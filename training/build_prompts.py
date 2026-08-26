"""
Emit one record per batch: the exact prompt the app would send, plus a
compact view of the same emails for labelling.

No LLM is called. summarize_emails builds the prompt as an f-string
inside the method, so the only way to get it byte-exact without editing
application code is to let it run and intercept the client -- here the
client returns a throwaway digest so nothing reaches Ollama.

    uv run python training/build_prompts.py

Writes training/data/_records.jsonl
"""

import asyncio
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from backend.llm.ollama import OllamaService        # noqa: E402
from backend.services.models import EmailMessage    # noqa: E402


RAW_FILE = REPO_ROOT / "training" / "data" / "raw" / "gmail_emails.json"
OUT_FILE = REPO_ROOT / "training" / "data" / "_records.jsonl"

# Matches get_recent_emails(limit=10) in the email worker.
EMAILS_PER_RECORD = 10

# Fractions of the *email* pool, not of the records.
SPLITS = {"train": 0.80, "validation": 0.10, "test": 0.10}

SEED = 20260825


def load_emails() -> list[EmailMessage]:
    if not RAW_FILE.exists():
        raise SystemExit(
            f"No raw emails at {RAW_FILE}. "
            "Run: uv run python -m backend.training.scrape_emails <user_id> 1500"
        )

    raw = json.loads(RAW_FILE.read_text(encoding="utf-8"))

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
    records. Splitting records instead leaked 100% of the previous
    test set into train.
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

STUB = json.dumps(
    {"summary": "", "priority_items": [], "action_items": []}
)


async def main():
    emails = load_emails()
    batches = split_by_email(emails)

    service = OllamaService()
    sent: dict = {}

    async def stub_chat(**kwargs):
        sent["messages"] = kwargs["messages"]
        return {"message": {"content": STUB}}

    service.client.chat = stub_chat

    written = 0

    with OUT_FILE.open("w", encoding="utf-8") as handle:
        for split, split_batches in batches.items():
            for index, batch in enumerate(split_batches):
                await service.summarize_emails(batch)

                record = {
                    "split": split,
                    "index": index,
                    "prompt": sent["messages"][0]["content"],
                    "emails": [
                        {
                            "sender": e.sender_name,
                            "from": e.sender_email,
                            "subject": e.subject,
                            "snippet": e.snippet[:220],
                            "when": e.timestamp,
                        }
                        for e in batch
                    ],
                }

                handle.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )

                written += 1

    print(f"wrote {written} records to {OUT_FILE}")
    print({k: len(v) for k, v in batches.items()})


if __name__ == "__main__":
    asyncio.run(main())
