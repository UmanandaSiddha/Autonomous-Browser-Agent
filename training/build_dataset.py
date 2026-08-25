"""
Build the email-digest fine-tuning dataset.

Teacher labels come from qwen3:8b via the exact serving path, so the
training text is byte-identical to what the app sends at inference.

    uv run python training/build_dataset.py [--limit N] [--resume]

Reads  training/data/raw/gmail_emails.json
Writes training/data/email_digest_{train,validation,test}.jsonl
"""

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from backend.llm.ollama import OllamaService          # noqa: E402
from backend.services.models import EmailMessage      # noqa: E402


RAW_FILE = REPO_ROOT / "training" / "data" / "raw" / "gmail_emails.json"
OUT_DIR = REPO_ROOT / "training" / "data"

# Matches get_recent_emails(limit=10) in the email worker.
EMAILS_PER_RECORD = 10

# Fractions of the *email* pool, not of the records.
SPLITS = {"train": 0.80, "validation": 0.10, "test": 0.10}

SEED = 20260825


def out_path(split: str) -> Path:
    return OUT_DIR / f"email_digest_{split}.jsonl"


def load_emails() -> list[EmailMessage]:
    if not RAW_FILE.exists():
        raise SystemExit(
            f"No raw emails at {RAW_FILE}\n"
            "Run: uv run python -m backend.training.scrape_emails <user_id> 1500"
        )

    raw = json.loads(RAW_FILE.read_text(encoding="utf-8"))

    # The scraper dedupes within a run; guard against merged files too.
    seen: set[str] = set()
    emails: list[EmailMessage] = []

    for item in raw:
        key = item.get("link") or item.get("thread_id")

        if key and key in seen:
            continue

        if key:
            seen.add(key)

        emails.append(EmailMessage(**item))

    return emails


def split_by_email(
    emails: list[EmailMessage],
) -> dict[str, list[list[EmailMessage]]]:
    """
    Assign every email to exactly one split BEFORE grouping into
    records. Splitting records instead is what leaked 100% of the
    previous test set into train.
    """

    shuffled = list(emails)
    random.Random(SEED).shuffle(shuffled)

    total = len(shuffled)
    n_train = int(total * SPLITS["train"])
    n_val = int(total * SPLITS["validation"])

    pools = {
        "train": shuffled[:n_train],
        "validation": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }

    batches = {}

    for split, pool in pools.items():
        # Drop the trailing partial batch: every record must have the
        # same shape the app sends.
        whole = len(pool) // EMAILS_PER_RECORD

        batches[split] = [
            pool[i * EMAILS_PER_RECORD:(i + 1) * EMAILS_PER_RECORD]
            for i in range(whole)
        ]

    return batches


async def label(service: OllamaService, batch: list[EmailMessage]):
    """
    Run one batch through the real serving path and capture the exact
    prompt that was sent. The prompt is an f-string inside
    summarize_emails and cannot be reached any other way without
    editing application code.
    """

    sent: dict = {}
    original_chat = service.client.chat

    async def capture(**kwargs):
        sent["messages"] = kwargs["messages"]
        return await original_chat(**kwargs)

    service.client.chat = capture

    try:
        digest = await service.summarize_emails(batch)
    finally:
        service.client.chat = original_chat

    return sent["messages"][0]["content"], digest


def already_done(path: Path) -> int:
    if not path.exists():
        return 0

    return sum(
        1 for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


async def main(limit: int | None, resume: bool):
    emails = load_emails()

    if limit:
        emails = emails[:limit]

    batches = split_by_email(emails)

    total = sum(len(v) for v in batches.values())

    print(f"emails       : {len(emails)}")
    print(f"records      : {total} " + str({k: len(v) for k, v in batches.items()}))
    print(f"per record   : {EMAILS_PER_RECORD}")
    print()

    # Truncate every split up front. Doing it per-split means a
    # crash mid-run leaves new data in one file and stale data in
    # another -- a mix that can look valid while being leaky.
    if not resume:
        for split in batches:
            out_path(split).write_text("", encoding="utf-8")

    service = OllamaService()
    done_count = 0
    started = time.monotonic()

    for split, split_batches in batches.items():
        path = out_path(split)

        skip = already_done(path) if resume else 0

        if skip:
            print(f"[{split}] resuming, {skip} already written")

        for index, batch in enumerate(split_batches):
            if index < skip:
                done_count += 1
                continue

            user_content, digest = await label(service, batch)

            record = {
                "messages": [
                    {"role": "user", "content": user_content},
                    {
                        "role": "assistant",
                        # mode="json" -- a plain dump leaves enums as
                        # Priority.HIGH rather than "high".
                        "content": json.dumps(
                            digest.model_dump(mode="json"),
                            ensure_ascii=False,
                        ),
                    },
                ]
            }

            # Append as we go: a 2h run should not lose everything to
            # one crash.
            with path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )

            done_count += 1
            elapsed = time.monotonic() - started
            rate = elapsed / max(done_count - skip, 1)
            left = (total - done_count) * rate

            print(
                f"[{split}] {index + 1}/{len(split_batches)} "
                f"({done_count}/{total} total) "
                f"items={len(digest.priority_items)} "
                f"eta={left / 60:.0f}m",
                flush=True,
            )

    print()
    print("done. now run: uv run python training/data/validate_dataset.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    asyncio.run(main(args.limit, args.resume))
