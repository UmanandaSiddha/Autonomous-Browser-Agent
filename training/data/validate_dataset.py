"""
Validate the email-digest dataset.

Structure alone is not enough -- the previous dataset was structurally
perfect and still unusable (100% train/test leakage, 14 distinct reason
strings across 304 items). These checks target the failures that
actually happened.

    uv run python training/data/validate_dataset.py
"""

import collections
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from backend.services.models import ActionType, EmailDigest   # noqa: E402


DATA_DIR = REPO_ROOT / "training" / "data"

SPLITS = ("train", "validation", "test")

EMAILS_PER_RECORD = 10

# The old dataset scored 4.6% here.
MIN_DISTINCT_REASONS = 0.60


def load(split: str) -> list[dict]:
    path = DATA_DIR / f"email_digest_{split}.jsonl"

    if not path.exists():
        raise SystemExit(f"Missing {path}. Run training/build_dataset.py")

    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def emails_in(record: dict) -> list[dict]:
    """
    Parse the email array out of the user message.

    summarize_emails sends only sender / sender_email / subject /
    snippet / timestamp -- it drops thread_id and link. An earlier
    version of this check looked for thread_id, found none, and
    happily reported "no leakage" on every record. Anything that
    cannot be parsed raises, because a silent empty set here is
    exactly how a leaky dataset passes review.
    """

    user = next(
        (
            m["content"]
            for m in record.get("messages", [])
            if m.get("role") == "user"
        ),
        None,
    )

    if user is None:
        raise ValueError("record has no user message")

    match = re.search(r"\[\s*\{.*\}\s*\]", user, re.S)

    if not match:
        raise ValueError("no email array found in the user message")

    return json.loads(match.group(0))


def email_keys(record: dict) -> set[tuple]:
    """
    Identity for leakage checks. thread_id is not available in the
    prompt, so key on the fields that are.
    """

    return {
        (
            email.get("sender_email"),
            email.get("subject"),
            email.get("timestamp"),
        )
        for email in emails_in(record)
    }


def main() -> int:
    failures: list[str] = []
    data = {split: load(split) for split in SPLITS}

    print("=== structure ===")

    digests: list[EmailDigest] = []

    for split, records in data.items():
        for number, record in enumerate(records, start=1):
            where = f"{split}:{number}"
            messages = record.get("messages", [])
            roles = [m["role"] for m in messages]

            # Must mirror what OllamaService.summarize_emails sends:
            # one user message, no system message.
            if roles != ["user", "assistant"]:
                failures.append(f"{where}: roles {roles}, expected user+assistant")
                continue

            try:
                count = len(emails_in(record))
            except Exception as exc:
                failures.append(f"{where}: cannot read emails -- {exc}")
                continue

            if count != EMAILS_PER_RECORD:
                failures.append(
                    f"{where}: {count} emails, expected {EMAILS_PER_RECORD}"
                )

            try:
                digest = EmailDigest.model_validate_json(
                    messages[1]["content"]
                )
            except Exception as exc:
                failures.append(f"{where}: assistant content invalid -- {exc}")
                continue

            digests.append(digest)

            derived = [
                item.action
                for item in digest.priority_items
                if item.action_type is ActionType.REQUIRED
            ]

            if digest.action_items != derived:
                failures.append(
                    f"{where}: action_items is not the required actions"
                )

        print(f"  {split:11} {len(records):4} records")

    print()
    print("=== leakage (the check the old dataset failed) ===")

    ids = {}

    for split, records in data.items():
        keys: set[tuple] = set()

        for number, record in enumerate(records, start=1):
            try:
                keys |= email_keys(record)
            except Exception as exc:
                failures.append(
                    f"{split}:{number}: cannot extract emails "
                    f"for the leakage check -- {exc}"
                )

        ids[split] = keys

    for a, b in (("train", "test"), ("train", "validation"), ("validation", "test")):
        shared = ids[a] & ids[b]
        status = "OK" if not shared else "FAIL"

        print(f"  {a:11} n {b:11} {len(shared):4} shared emails  [{status}]")

        if shared:
            failures.append(
                f"{len(shared)} emails appear in both {a} and {b}"
            )

    print(f"  distinct emails total: {sum(len(v) for v in ids.values())}")

    print()
    print("=== label diversity ===")

    reasons = [i.reason for d in digests for i in d.priority_items]
    summaries = [re.sub(r"\d+", "N", d.summary) for d in digests]

    if reasons:
        ratio = len(set(reasons)) / len(reasons)

        print(
            f"  reasons  : {len(set(reasons))}/{len(reasons)} distinct "
            f"({ratio:.0%})"
        )

        if ratio < MIN_DISTINCT_REASONS:
            failures.append(
                f"reasons only {ratio:.0%} distinct, "
                f"need {MIN_DISTINCT_REASONS:.0%} -- labels look templated"
            )

        top, n = collections.Counter(reasons).most_common(1)[0]
        print(f"  most repeated reason: {n}x  {top[:60]}")

    print(f"  summaries: {len(set(summaries))}/{len(summaries)} distinct shapes")

    print()
    print("=== distribution (reported, not asserted) ===")

    priorities = collections.Counter(
        i.priority.value for d in digests for i in d.priority_items
    )
    types = collections.Counter(
        i.action_type.value for d in digests for i in d.priority_items
    )
    per_record = collections.Counter(len(d.priority_items) for d in digests)

    print(f"  priority   : {dict(priorities)}")
    print(f"  action_type: {dict(types)}")
    print(f"  items/record: {dict(sorted(per_record.items()))}")

    if digests:
        flagged = sum(len(d.priority_items) for d in digests)
        slots = len(digests) * EMAILS_PER_RECORD
        print(f"  flag rate  : {flagged}/{slots} ({100 * flagged / slots:.0f}% of emails)")

    print()

    if failures:
        print(f"FAILED ({len(failures)})")

        for line in failures[:15]:
            print(f"  - {line}")

        if len(failures) > 15:
            print(f"  ... and {len(failures) - 15} more")

        return 1

    print("PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
