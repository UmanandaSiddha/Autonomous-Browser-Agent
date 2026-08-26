"""
Merge the LoRA adapter into the base model and register it with Ollama.

Ollama 0.32+ imports Qwen2-architecture safetensors directly, so no
GGUF conversion or llama.cpp build is needed.

    uv run python training/export_model.py
    ollama create email-digest -f training/outputs/merged/Modelfile

Runs on CPU on purpose: merging needs the base in fp16 (~3.1 GB) and
4-bit weights cannot be merged, so this would not fit in 4 GB of VRAM.
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
ADAPTER = REPO_ROOT / "training" / "outputs" / "lora-email-digest" / "adapter"
MERGED = REPO_ROOT / "training" / "outputs" / "merged"

MODELFILE = """FROM .

# Matches backend/llm/ollama.py: greedy, deterministic digests.
PARAMETER temperature 0
PARAMETER num_ctx 4096
"""


def _make_ollama_compatible(out_dir: Path, base_model: str):
    """
    transformers 5.x writes a config schema Ollama 0.32 cannot read.
    It relocated rope_theta into a rope_parameters object and renamed
    torch_dtype to dtype. Ollama still looks for the old top-level
    keys, finds nothing, falls back to a default RoPE frequency and
    the model emits pure noise -- correct weights, wrong positions.

    It also needs the legacy BPE vocab files, which save_pretrained
    omits when it writes the fast tokenizer.
    """

    config_path = out_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    rope = config.get("rope_parameters") or {}

    restored = []

    if not config.get("rope_theta") and rope.get("rope_theta"):
        config["rope_theta"] = rope["rope_theta"]
        restored.append("rope_theta")

    if not config.get("torch_dtype") and config.get("dtype"):
        config["torch_dtype"] = config["dtype"]
        restored.append("torch_dtype")

    config_path.write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    print(f"[COMPAT] restored legacy config keys: {restored}")

    # vocab.json / merges.txt for the converter's BPE vocabulary
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    written = tokenizer.save_vocabulary(str(out_dir))

    print(
        "[COMPAT] wrote vocab files: "
        f"{[Path(f).name for f in written]}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default=str(ADAPTER))
    args = ap.parse_args()

    adapter_dir = Path(args.adapter)

    if not adapter_dir.exists():
        raise SystemExit(
            f"No adapter at {adapter_dir}. Train first."
        )

    print(f"[LOAD] base {args.model} in fp16 on CPU")

    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16,
        device_map="cpu",
    )

    print(f"[LOAD] adapter {adapter_dir}")

    model = PeftModel.from_pretrained(base, str(adapter_dir))

    print("[MERGE] folding adapter weights into the base")

    model = model.merge_and_unload()

    if MERGED.exists():
        shutil.rmtree(MERGED)

    MERGED.mkdir(parents=True)

    model.save_pretrained(str(MERGED), safe_serialization=True)

    # Ollama reads the chat template out of tokenizer_config.json, so
    # the tokenizer has to travel with the weights.
    AutoTokenizer.from_pretrained(args.model).save_pretrained(
        str(MERGED)
    )

    _make_ollama_compatible(MERGED, args.model)

    (MERGED / "Modelfile").write_text(MODELFILE, encoding="utf-8")

    size = sum(
        f.stat().st_size for f in MERGED.rglob("*") if f.is_file()
    )

    print()
    print(f"[DONE] merged model at {MERGED} ({size / 1e9:.2f} GB)")
    print()
    print("Next:")
    print(f"  cd {MERGED}")
    print("  ollama create email-digest -f Modelfile")
    print("  ollama run email-digest 'hi'")


if __name__ == "__main__":
    main()
