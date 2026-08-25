import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from backend.agent.graph import build_graph
from backend.browser.auth import GmailAuth
from backend.browser.gmail import GmailService
from backend.browser.manager import BrowserManager

from backend.db.database import SessionLocal
from backend.db.models import JobStatus, JobStep

from backend.jobs.manager import job_manager
from backend.services.models import EmailDigest, EmailMessage


def _write_digest(
    user_id: str,
    job_id: str,
    digest: EmailDigest,
    emails: list[EmailMessage],
) -> Path:
    """
    Persist the digest as Markdown at outputs/<user_id>/<job_id>.md
    """

    path = Path("outputs") / user_id / f"{job_id}.md"

    path.parent.mkdir(parents=True, exist_ok=True)

    generated = datetime.now(timezone.utc).strftime(
        "%Y-%m-%d %H:%M UTC"
    )

    lines = [
        "# Email Digest",
        "",
        f"- **Job:** `{job_id}`",
        f"- **Generated:** {generated}",
        "",
        "## Summary",
        "",
        digest.summary,
        "",
    ]

    if digest.priority_items:
        lines += ["## Needs Attention", ""]

        for item in digest.priority_items:
            lines += [
                f"### {item.action}",
                "",
                f"`{item.priority.value}` | `{item.action_type.value}`",
                "",
                item.reason,
                "",
            ]

    if digest.action_items:
        lines += ["## Action Items", ""]
        lines += [f"- [ ] {a}" for a in digest.action_items]
        lines += [""]

    # Every email that was read, so the digest above can be
    # checked against what it actually saw.
    lines += [f"## Emails Reviewed ({len(emails)})", ""]

    for email in emails:
        subject = email.subject or "(no subject)"

        title = (
            f"[{subject}]({email.link})"
            if email.link
            else subject
        )

        sender = email.sender_name or email.sender_email

        lines += [
            f"- {title}",
            f"  {sender} - {email.timestamp}",
        ]

    lines += [""]

    lines += [
        "## Raw Digest",
        "",
        "```json",
        json.dumps(
            digest.model_dump(mode="json"),
            indent=2,
            ensure_ascii=False,
        ),
        "```",
        "",
    ]

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return path


def _record_failure(db, job_id: str, message: str):
    """
    Mark a job failed. Must never raise: this is the only
    thing standing between a crash and a job stuck on
    "running" forever.
    """
    try:
        db.rollback()

        job_manager.update_job(
            db,
            job_id,
            status=JobStatus.FAILED,
            progress=100,
            error=message,
        )

    except Exception as exc:
        print(
            f"[EMAIL WORKER] Could not record "
            f"failure for {job_id}: {exc}"
        )


async def run_email_automation(
    job_id: str,
    user_id: str
):
    browser = BrowserManager(user_id)

    # This runs after the HTTP response is gone, so the
    # request-scoped session is already closed. Own one.
    db = SessionLocal()

    try:
        # -------------------------------------------------
        # 1. START
        # -------------------------------------------------

        print(
            f"[EMAIL WORKER] Starting {job_id}"
        )

        job_manager.update_job(
            db,
            job_id,
            status=JobStatus.RUNNING,
            step=JobStep.AUTHENTICATING,
            progress=10,
        )

        # -------------------------------------------------
        # 2. OPEN BROWSER
        # -------------------------------------------------

        page = await browser.launch(
            headless=True
        )

        print(
            "[EMAIL WORKER] Browser ready"
        )

        # -------------------------------------------------
        # 3. AUTH CHECK
        # -------------------------------------------------

        auth = GmailAuth()

        authenticated = (
            await auth.is_authenticated(page)
        )

        if not authenticated:
            raise RuntimeError(
                "Gmail is not authenticated."
            )

        print(
            "[EMAIL WORKER] Gmail authenticated"
        )

        # -------------------------------------------------
        # 4. EXTRACT EMAILS
        # -------------------------------------------------

        job_manager.update_job(
            db,
            job_id,
            step=JobStep.EXTRACTING_EMAILS,
            progress=30,
        )

        print(
            "[EMAIL WORKER] Extracting emails..."
        )

        gmail = GmailService(page)

        await gmail.open_inbox()

        emails = await gmail.get_recent_emails(
            limit=10
        )

        print(
            f"[EMAIL WORKER] Extracted "
            f"{len(emails)} emails"
        )

        # Fail here rather than inside the graph: an empty
        # inbox is deterministic, so retrying it just burns
        # the retry budget on the same result.
        if not emails:
            raise RuntimeError(
                "No emails found in the inbox."
            )

        # -------------------------------------------------
        # 5. LANGGRAPH
        # -------------------------------------------------

        job_manager.update_job(
            db,
            job_id,
            step=JobStep.SUMMARIZING,
            progress=50,
        )

        print(
            "[EMAIL WORKER] Starting LangGraph..."
        )

        graph = build_graph()

        initial_state = {
            "authenticated": authenticated,
            "emails": emails,
            "digest": None,
            "error": None,
            "retry_count": 0,
        }

        print(
            "[EMAIL WORKER] "
            "Invoking LangGraph"
        )

        result = await graph.ainvoke(
            initial_state
        )

        # -------------------------------------------------
        # 6. VALIDATION
        # -------------------------------------------------

        job_manager.update_job(
            db,
            job_id,
            step=JobStep.VALIDATING,
            progress=90,
        )

        if result.get("error"):
            raise RuntimeError(
                result["error"]
            )

        digest = result.get("digest")

        if digest is None:
            raise RuntimeError(
                "Agent produced no digest."
            )

        # -------------------------------------------------
        # 7. PERSIST + COMPLETE
        # -------------------------------------------------

        digest_path = _write_digest(
            user_id,
            job_id,
            digest,
            emails,
        )

        print(
            f"[EMAIL WORKER] Digest written to "
            f"{digest_path}"
        )

        job_manager.update_job(
            db,
            job_id,
            status=JobStatus.COMPLETED,
            step=JobStep.COMPLETED,
            progress=100,
            result=digest.model_dump(mode="json"),
        )

        print(
            f"[EMAIL WORKER] "
            f"{job_id} completed successfully"
        )

    except asyncio.CancelledError:
        # Shutdown cancels in-flight tasks. CancelledError is not
        # an Exception, so without this the job stays "running".
        print(
            f"[EMAIL WORKER] "
            f"{job_id} cancelled"
        )

        _record_failure(
            db,
            job_id,
            "Automation was cancelled.",
        )

        raise

    except Exception as exc:

        print(
            f"[EMAIL WORKER] "
            f"{job_id} failed: {exc}"
        )

        _record_failure(db, job_id, str(exc))

    finally:

        await browser.close()

        db.close()

        print(
            f"[EMAIL WORKER] "
            f"{job_id} browser closed"
        )
