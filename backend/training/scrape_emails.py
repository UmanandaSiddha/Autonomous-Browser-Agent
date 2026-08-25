"""
Scrape real Gmail messages to build a fine-tuning corpus.

Run from anywhere -- paths are anchored to the repo root:

    uv run python -m backend.training.scrape_emails <user_id> [limit]

The user_id must match a connected profile under browser_profiles/.
"""

import asyncio
import json
import sys
from pathlib import Path

from backend.browser.gmail import GmailService
from backend.browser.manager import BrowserManager


# Anchored to the repo root so the script works from any directory.
REPO_ROOT = Path(__file__).resolve().parents[2]

PROFILES_DIR = REPO_ROOT / "browser_profiles"

OUTPUT_FILE = (
    REPO_ROOT / "training" / "data" / "raw" / "gmail_emails.json"
)

DEFAULT_LIMIT = 100


def resolve_user_id(argv: list[str]) -> str:
    """
    Take the user id from argv, or fall back to the only connected
    profile. Guessing between several would scrape the wrong inbox.
    """

    if len(argv) > 1:
        return argv[1]

    if not PROFILES_DIR.exists():
        raise SystemExit(
            f"No profiles found at {PROFILES_DIR}. "
            "Connect Gmail through /api/browser-auth/gmail/connect first."
        )

    profiles = sorted(
        p.name for p in PROFILES_DIR.iterdir() if p.is_dir()
    )

    if not profiles:
        raise SystemExit(
            f"No profiles found at {PROFILES_DIR}. "
            "Connect Gmail through /api/browser-auth/gmail/connect first."
        )

    if len(profiles) > 1:
        raise SystemExit(
            "Several profiles exist, pass the one to scrape:\n  "
            + "\n  ".join(profiles)
        )

    return profiles[0]


async def scrape_emails(user_id: str, limit: int):
    print("=" * 70)
    print("GMAIL TRAINING DATA SCRAPER")
    print("=" * 70)
    print(f"[CONFIG] user_id : {user_id}")
    print(f"[CONFIG] limit   : {limit}")
    print(f"[CONFIG] output  : {OUTPUT_FILE}")

    browser = BrowserManager(user_id)

    print(f"[CONFIG] profile : {browser.profile_dir.resolve()}")

    if not browser.profile_dir.exists():
        raise SystemExit(
            f"\nNo browser profile at {browser.profile_dir.resolve()}\n"
            "That user has not connected Gmail yet."
        )

    try:
        print("[BROWSER] Launching Camoufox...")

        page = await browser.launch(headless=True)

        print("[BROWSER] Browser ready")

        gmail = GmailService(page)

        print("[GMAIL] Opening inbox...")

        await gmail.open_inbox()

        print("[GMAIL] Inbox loaded")

        print(f"[GMAIL] Extracting up to {limit} emails...")

        emails = await gmail.get_recent_emails(limit=limit)

        print(f"[GMAIL] Extracted {len(emails)} emails")

        if len(emails) < limit:
            print(
                f"[GMAIL] Note: asked for {limit}, got {len(emails)} "
                "-- ran out of inbox pages."
            )

        if not emails:
            raise SystemExit(
                "Nothing extracted. The session is probably no "
                "longer authenticated."
            )

        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

        with OUTPUT_FILE.open("w", encoding="utf-8") as file:
            json.dump(
                [email.model_dump() for email in emails],
                file,
                indent=2,
                ensure_ascii=False,
            )

        print(f"[DATA] Saved {len(emails)} emails to: {OUTPUT_FILE}")

        print()
        print("=" * 70)
        print("SCRAPE COMPLETE")
        print("=" * 70)

        for index, email in enumerate(emails[:5], start=1):
            print()
            print(f"EMAIL {index}")
            print(f"Sender   : {email.sender_name}")
            print(f"Email    : {email.sender_email}")
            print(f"Subject  : {email.subject}")
            print(f"Timestamp: {email.timestamp}")

        print()
        print(f"Total emails saved: {len(emails)}")

    except Exception as exc:
        print()
        print("[ERROR] Email scraping failed")
        print(f"[ERROR] {type(exc).__name__}: {exc}")

        raise

    finally:
        print()
        print("[BROWSER] Closing browser...")

        await browser.close()

        print("[BROWSER] Browser closed")


if __name__ == "__main__":
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_LIMIT

    asyncio.run(
        scrape_emails(resolve_user_id(sys.argv), limit)
    )
