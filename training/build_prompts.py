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
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from backend.llm.ollama import OllamaService              # noqa: E402
from training.build_dataset import load_emails, split_by_email  # noqa: E402


OUT_FILE = REPO_ROOT / "training" / "data" / "_records.jsonl"

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
