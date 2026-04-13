import os
import pickle
import hashlib
import argparse
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── Defaults ────────────────────────────────────────────────────────────────────
TASKS = [
    "Summarization", "Creative writing", "Brainstorming",
    "Closed QA", "Classification", "Open QA", "Multiple choice",
]

# RTX 4090 (24 GB VRAM).
# BATCH_SIZE_BY_TASK = {
#     "Summarization":    2,   # ~4096 tok input → ~1.3 GB/seq
#     "Closed QA":        4,   # ~1024 tok input → ~440 MB/seq
#     "Creative writing": 4,
#     "Brainstorming":    4,
#     "Open QA":          8,   # ~512 tok input  → ~294 MB/seq
#     "Classification":   8,
#     "Multiple choice":  8,
# }
#A100 80GB
BATCH_SIZE_BY_TASK = {
    "Summarization":   16,   # 2 × 11 → 16 (conservative)
    "Closed QA":       32,   # 4 × 11 → 32
    "Creative writing": 32,
    "Brainstorming":   32,
    "Open QA":         64,   # 8 × 11 → 64
    "Classification":  64,
    "Multiple choice": 64,
}

SYSTEM_PROMPT = (
    "Below is an instruction that describes a task, paired with an input that provides "
    "further context. Write a response that appropriately completes the request."
)


# ── Helpers ─────────────────────────────────────────────────────────────────────
def generate_id(question, context, category):
    hash_input = f"{question}_{context}_{category}"
    return hashlib.sha256(hash_input.encode()).hexdigest()


def build_messages(row, few_shot_examples):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in few_shot_examples.itertuples(index=False):
        user_content = ex.question if pd.isna(ex.context) else f"{ex.question}\n{ex.context}"
        messages.append({"role": "user",      "content": user_content})
        messages.append({"role": "assistant", "content": ex.answer})
    user_content = row["question"] if pd.isna(row["context"]) else f"{row['question']}\n{row['context']}"
    messages.append({"role": "user", "content": user_content})
    return messages


@torch.inference_mode()
def generate_batch(model, tokenizer, batch_messages, max_new_tokens, temperature, top_p, top_k, repetition_penalty):
    """Run inference on a batch of message lists simultaneously."""
    texts = [
        tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for msgs in batch_messages
    ]
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
    )

    input_len = inputs["input_ids"].shape[-1]
    return [
        tokenizer.decode(out[input_len:], skip_special_tokens=True).strip()
        for out in output_ids
    ]


# ── Evaluation ──────────────────────────────────────────────────────────────────
def evaluate_task(task_type, all_eval, model, tokenizer, args):
    batch_size = args.batch_size_override if args.batch_size_override else BATCH_SIZE_BY_TASK.get(task_type, 4)

    eval_df = all_eval[
        (all_eval["task_type"] == task_type) &
        (all_eval["thai_specific"] == "YES")
    ][["question", "context", "answer"]].copy().reset_index(drop=True)

    eval_df["id"] = eval_df.apply(
        lambda r: generate_id(r["question"], r["context"], task_type), axis=1
    )

    if args.test:
        eval_df = eval_df.sample(min(20, len(eval_df)), random_state=42)
        print(f"[TEST] {task_type}: using {len(eval_df)} samples")

    if len(eval_df) == 0:
        print(f"[SKIP] {task_type}: no thai_specific='YES' rows found")
        return None

    # checkpoint
    slug = task_type.lower().replace(" ", "_")
    ckpt_path = os.path.join(args.checkpoint_dir, f"{slug}_checkpoint.pkl")
    predictions = {}

    if os.path.exists(ckpt_path):
        with open(ckpt_path, "rb") as f:
            ckpt = pickle.load(f)
        if isinstance(ckpt.get("predictions"), dict):
            predictions = ckpt["predictions"]
        print(f"[RESUME] {task_type}: {len(predictions)} cached predictions")

    remaining = [rid for rid in eval_df["id"] if rid not in predictions]

    # chunk into batches
    batches = [remaining[i:i + batch_size] for i in range(0, len(remaining), batch_size)]

    for batch_ids in tqdm(batches, desc=f"{task_type} (batch={batch_size})"):
        batch_messages = []
        for row_id in batch_ids:
            row  = eval_df[eval_df["id"] == row_id].iloc[0]
            pool = eval_df[eval_df["question"] != row["question"]]
            few_shot_examples = pool.sample(min(args.few_shot, len(pool)), random_state=None)
            batch_messages.append(build_messages(row, few_shot_examples))

        try:
            batch_preds = generate_batch(
                model, tokenizer, batch_messages,
                args.max_new_tokens, args.temperature,
                args.top_p, args.top_k, args.repetition_penalty,
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and batch_size > 1:
                # OOM fallback: retry each item one-by-one
                print(f"\n[OOM] {task_type} batch={batch_size} → retrying one-by-one")
                torch.cuda.empty_cache()
                batch_preds = []
                for msgs in batch_messages:
                    try:
                        batch_preds.extend(generate_batch(
                            model, tokenizer, [msgs],
                            args.max_new_tokens, args.temperature,
                            args.top_p, args.top_k, args.repetition_penalty,
                        ))
                    except Exception as e2:
                        print(f"  Single-item error: {e2}")
                        batch_preds.append(None)
            else:
                print(f"Batch error: {e}")
                batch_preds = [None] * len(batch_ids)

        for row_id, pred in zip(batch_ids, batch_preds):
            if pred is not None:
                predictions[row_id] = pred

        with open(ckpt_path, "wb") as f:
            pickle.dump({"predictions": predictions}, f)

    eval_df["prediction"] = eval_df["id"].map(predictions)
    return eval_df


def save_result(task_type, result_df, output_dir):
    slug = task_type.lower().replace(" ", "_")
    out_path = os.path.join(output_dir, f"{slug}_yes_eval.csv")
    result_df.to_csv(out_path, index=False)
    print(f"Saved → {out_path}")


# ── Main ────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Evaluate SeaLLM (HuggingFace) on Thai-specific subset")
    parser.add_argument("--model-name",         type=str,   default="SeaLLMs/SeaLLMs-v3-7B-Chat")
    parser.add_argument("--eval-csv",           type=str,   default="./eval/eval_set.csv")
    parser.add_argument("--output-dir",         type=str,   default="./eval/sea_llm_results")
    parser.add_argument("--checkpoint-dir",     type=str,   default="./checkpoints_sea_llm")
    parser.add_argument("--few-shot",           type=int,   default=0)
    parser.add_argument("--max-new-tokens",     type=int,   default=512)
    parser.add_argument("--temperature",        type=float, default=0.05)
    parser.add_argument("--top-p",              type=float, default=0.9)
    parser.add_argument("--top-k",              type=int,   default=50)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--batch-size",         type=int,   default=0,
                        dest="batch_size_override",
                        help="Override per-task batch sizes (0 = use defaults from BATCH_SIZE_BY_TASK)")
    parser.add_argument("--test",               action="store_true", help="Use only 20 samples per task")
    args = parser.parse_args()

    os.makedirs(args.output_dir,     exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print(f"\nLoading {args.model_name} …")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    # left-padding required for batched generation
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print("Model loaded.\n")

    all_eval = pd.read_csv(args.eval_csv)
    all_eval = all_eval.rename(columns={"Instruction": "question", "Input": "context", "Output": "answer"})
    print(f"Dataset: {len(all_eval)} rows total")
    print(f"thai_specific counts:\n{all_eval['thai_specific'].value_counts()}\n")

    print("Batch sizes per task:")
    for task in TASKS:
        bs = args.batch_size_override if args.batch_size_override else BATCH_SIZE_BY_TASK.get(task, 4)
        print(f"  {task:25s}: {bs}")
    print()

    results = {}
    for task in TASKS:
        bs = args.batch_size_override if args.batch_size_override else BATCH_SIZE_BY_TASK.get(task, 4)
        print(f"\n{'='*60}\nTask: {task}  (batch_size={bs})")
        df_result = evaluate_task(task, all_eval, model, tokenizer, args)
        if df_result is not None:
            results[task] = df_result
            save_result(task, df_result, args.output_dir)

    print("\n── Summary ─────────────────────────────────────────────────")
    for task, df in results.items():
        bs = args.batch_size_override if args.batch_size_override else BATCH_SIZE_BY_TASK.get(task, 4)
        predicted = df["prediction"].notna().sum()
        print(f"  {task:25s} (batch={bs}): {predicted}/{len(df)} predictions completed")
    print("All tasks done.")


if __name__ == "__main__":
    main()
