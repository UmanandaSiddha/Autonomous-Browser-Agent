"""
Join the extracted prompts with hand-written labels into the final
train/validation/test JSONL files.

    uv run python training/assemble_dataset.py

Reads  training/data/_records.jsonl   (prompts, from build_prompts.py)
       training/data/_labels.jsonl    (labels, hand-written)
Writes training/data/email_digest_{train,validation,test}.jsonl
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from backend.services.models import ActionType, EmailDigest   # noqa: E402


DATA = REPO_ROOT / "training" / "data"
RECORDS = DATA / "_records.jsonl"
LABELS = DATA / "_labels.jsonl"


def read(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"missing {path}")

    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main():
    records = {(r["split"], r["index"]): r for r in read(RECORDS)}
    labels = read(LABELS)

    by_split: dict[str, list[str]] = {}
    problems: list[str] = []

    for label in labels:
        key = (label["split"], label["index"])

        if key not in records:
            problems.append(f"{key}: no matching prompt")
            continue

        digest = EmailDigest.model_validate(label["digest"])

        # action_items is derived at serving time, so derive it here
        # too rather than trusting whatever was typed.
        digest.action_items = [
            item.action
            for item in digest.priority_items
            if item.action_type is ActionType.REQUIRED
        ]

        by_split.setdefault(label["split"], []).append(
            json.dumps(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": records[key]["prompt"],
                        },
                        {
                            "role": "assistant",
                            "content": json.dumps(
                                digest.model_dump(mode="json"),
                                ensure_ascii=False,
                            ),
                        },
                    ]
                },
                ensure_ascii=False,
            )
        )

    for split in ("train", "validation", "test"):
        path = DATA / f"email_digest_{split}.jsonl"
        lines = by_split.get(split, [])

        path.write_text(
            "\n".join(lines) + ("\n" if lines else ""),
            encoding="utf-8",
        )

        total = sum(1 for k in records if k[0] == split)

        print(f"  {split:11} {len(lines):4} / {total} labelled")

    if problems:
        print()
        print("problems:")

        for line in problems:
            print(f"  - {line}")

        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
