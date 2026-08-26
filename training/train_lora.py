"""
QLoRA fine-tune for the email digest task.

The base model is frozen and loaded in 4-bit; only the LoRA adapters
train. Sized for a 4 GB card.

    uv run python training/train_lora.py
    uv run python training/train_lora.py --model Qwen/Qwen2.5-0.5B-Instruct

Stop Ollama first -- it holds ~2.3 GB of the 4 GB.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "training" / "data"
OUT = REPO_ROOT / "training" / "outputs" / "lora-email-digest"


def load_split(split: str, tokenizer) -> Dataset:
    """
    Pre-tokenise into input_ids + completion_mask.

    TRL 0.19.1 with transformers 5.15 mangles conversational
    prompt-completion data -- it hands the collator an input_ids that
    is a nested dict and a completion_mask of length 2, which dies as
    "Could not infer dtype of dict". Tokenising here sidesteps that
    path entirely and makes the prompt masking explicit and checkable.
    """

    path = DATA / f"email_digest_{split}.jsonl"

    if not path.exists():
        raise SystemExit(f"missing {path}")

    rows = []

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue

        messages = json.loads(line)["messages"]
        user = [m for m in messages if m["role"] == "user"]
        answer = next(
            m["content"] for m in messages if m["role"] == "assistant"
        )

        prompt_text = tokenizer.apply_chat_template(
            user,
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = prompt_text + answer + tokenizer.eos_token

        prompt_ids = tokenizer(
            prompt_text, add_special_tokens=False
        )["input_ids"]
        full_ids = tokenizer(
            full_text, add_special_tokens=False
        )["input_ids"]

        # If the prompt is not a clean prefix the mask would silently
        # train on the wrong tokens, so fail loudly instead.
        if full_ids[:len(prompt_ids)] != prompt_ids:
            raise SystemExit(
                "Tokenised prompt is not a prefix of prompt+completion; "
                "prompt masking would be wrong."
            )

        rows.append(
            {
                "input_ids": full_ids,
                "completion_mask": (
                    [0] * len(prompt_ids)
                    + [1] * (len(full_ids) - len(prompt_ids))
                ),
            }
        )

    return Dataset.from_list(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--max-seq", type=int, default=4096)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--accum", type=int, default=8)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue from the newest checkpoint in the output dir",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA device visible.")

    free, total = torch.cuda.mem_get_info()

    print(f"[GPU] {torch.cuda.get_device_name(0)}")
    print(f"[GPU] free {free / 1e9:.2f} / {total / 1e9:.2f} GB")

    if free < 3.4e9:
        print(
            "[GPU] WARNING: less than 3.4 GB free. Stop Ollama "
            "(it holds ~2.3 GB) or this will OOM."
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = load_split("train", tokenizer)
    eval_ds = load_split("validation", tokenizer)

    trained = sum(sum(r) for r in train_ds["completion_mask"])
    tokens = sum(len(r) for r in train_ds["input_ids"])

    print(f"[DATA] train {len(train_ds)}  validation {len(eval_ds)}")
    print(
        f"[DATA] loss on {trained:,} of {tokens:,} tokens "
        f"({100 * trained / tokens:.0f}%) -- prompt is masked"
    )

    # Frozen 4-bit base. Only the adapters below are trainable.
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    peft_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    config = SFTConfig(
        output_dir=str(OUT),
        model_init_kwargs={
            "quantization_config": quant,
            "dtype": torch.bfloat16,
            "device_map": {"": 0},
        },
        max_length=args.max_seq,
        packing=False,
        completion_only_loss=True,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        # transformers 5.x dropped warmup_ratio; 60 records at
        # accum 8 is ~7 steps an epoch, so 2 steps is the same 3%.
        warmup_steps=2,
        # 90 records overfits fast; keep the regularisation on.
        weight_decay=0.01,
        max_grad_norm=0.3,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        logging_steps=1,
        # Step-based, not epoch-based: an epoch is ~41 min here, so
        # a crash between epoch boundaries loses the whole lot.
        # Every 4 steps caps the loss at ~23 min.
        eval_strategy="steps",
        eval_steps=4,
        save_strategy="steps",
        save_steps=4,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=[],
        seed=20260826,
    )

    trainer = SFTTrainer(
        model=args.model,
        args=config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainable = sum(
        p.numel() for p in trainer.model.parameters() if p.requires_grad
    )
    total_p = sum(p.numel() for p in trainer.model.parameters())

    print(
        f"[LORA] trainable {trainable:,} / {total_p:,} "
        f"({100 * trainable / total_p:.2f}%) -- base is frozen"
    )

    resume = False

    if args.resume:
        checkpoints = sorted(
            OUT.glob("checkpoint-*"),
            key=lambda d: int(d.name.split("-")[1]),
        )

        if checkpoints:
            resume = str(checkpoints[-1])
            print(f"[RESUME] continuing from {checkpoints[-1].name}")
        else:
            print("[RESUME] no checkpoint found, starting fresh")

    trainer.train(resume_from_checkpoint=resume or None)

    adapter_dir = OUT / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    print()
    print(f"[DONE] adapter saved to {adapter_dir}")
    print("[NEXT] uv run python training/evaluate.py")


if __name__ == "__main__":
    main()
