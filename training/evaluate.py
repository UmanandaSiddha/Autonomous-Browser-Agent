"""
Score a model against the hand-labelled test split.

Run it twice to get the number that matters -- the delta:

    uv run python training/evaluate.py                      # base model
    uv run python training/evaluate.py --adapter training/outputs/lora-email-digest/adapter

These metrics measure FORMAT and CALIBRATION (does it emit valid JSON,
does it flag a sensible number of emails, does it stay quiet on a batch
with nothing in it). They do not measure whether the right emails were
picked -- read --show samples for that.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from backend.services.models import EmailDigest  # noqa: E402

TEST_FILE = REPO_ROOT / "training" / "data" / "email_digest_test.jsonl"


def load_test() -> list[dict]:
    return [
        json.loads(line)
        for line in TEST_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build(model_id: str, adapter: str | None):
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant,
        dtype=torch.bfloat16,
        device_map={"": 0},
    )

    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
        print(f"[MODEL] {model_id} + adapter {adapter}")
    else:
        print(f"[MODEL] {model_id} (base, no adapter)")

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(adapter or model_id)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def generate(model, tokenizer, prompt_messages, max_new_tokens):
    text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    return tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()


def parse(raw: str) -> EmailDigest | None:
    """Tolerate a model that wraps its JSON in prose or fences."""

    candidate = raw

    if "```" in candidate:
        parts = candidate.split("```")
        candidate = max(parts, key=len).removeprefix("json").strip()

    start, end = candidate.find("{"), candidate.rfind("}")

    if start == -1 or end == -1:
        return None

    try:
        return EmailDigest.model_validate_json(candidate[start:end + 1])
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=700)
    ap.add_argument("--show", type=int, default=2)
    args = ap.parse_args()

    records = load_test()
    model, tokenizer = build(args.model, args.adapter)

    valid = 0
    abs_err = []
    pred_items = 0
    ref_items = 0
    empty_correct = 0
    empty_total = 0
    samples = []

    for i, record in enumerate(records):
        prompt = [
            m for m in record["messages"] if m["role"] == "user"
        ]
        reference = EmailDigest.model_validate_json(
            next(
                m["content"]
                for m in record["messages"]
                if m["role"] == "assistant"
            )
        )

        raw = generate(model, tokenizer, prompt, args.max_new_tokens)
        prediction = parse(raw)

        n_ref = len(reference.priority_items)
        ref_items += n_ref

        if prediction is None:
            print(f"  [{i:2}] INVALID JSON")
            abs_err.append(n_ref)
        else:
            valid += 1
            n_pred = len(prediction.priority_items)
            pred_items += n_pred
            abs_err.append(abs(n_pred - n_ref))
            print(f"  [{i:2}] items pred={n_pred} ref={n_ref}")

        if n_ref == 0:
            empty_total += 1

            if prediction is not None and not prediction.priority_items:
                empty_correct += 1

        if len(samples) < args.show:
            samples.append((reference, prediction, raw))

    n = len(records)

    print()
    print("=" * 58)
    print(f"  valid JSON        {valid}/{n} ({100 * valid / n:.0f}%)")
    print(f"  item count MAE    {sum(abs_err) / n:.2f}")
    print(f"  items predicted   {pred_items} (reference {ref_items})")

    if empty_total:
        print(
            f"  quiet on empty    {empty_correct}/{empty_total} "
            "batches with nothing to flag"
        )

    print(f"  flag rate         {pred_items}/{n * 10} "
          f"({100 * pred_items / (n * 10):.0f}% of emails, "
          f"reference {100 * ref_items / (n * 10):.0f}%)")
    print("=" * 58)

    for reference, prediction, raw in samples:
        print()
        print("REFERENCE:", reference.summary[:100])

        if prediction is None:
            print("PREDICTED: <unparseable>", raw[:120])
        else:
            print("PREDICTED:", prediction.summary[:100])


if __name__ == "__main__":
    main()
