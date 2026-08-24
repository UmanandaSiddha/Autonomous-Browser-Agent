from backend.agent.graph import build_graph
from backend.browser.auth import GmailAuth
from backend.browser.gmail import GmailService
from backend.browser.manager import BrowserManager

from backend.jobs.manager import job_manager
from backend.jobs.models import (
    JobStatus,
    JobStep,
)


async def run_email_automation(
    job_id: str,
):
    browser = BrowserManager()

    try:
        # -------------------------------------------------
        # 1. START
        # -------------------------------------------------

        print(
            f"[EMAIL WORKER] Starting {job_id}"
        )

        job_manager.update_job(
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

        # -------------------------------------------------
        # 5. LANGGRAPH
        # -------------------------------------------------

        job_manager.update_job(
            job_id,
            step=JobStep.SUMMARIZING,
            progress=50,
        )

        print(
            "[EMAIL WORKER] Starting LangGraph..."
        )

        graph = build_graph()

        result = await graph.ainvoke(
            {
                "authenticated": True,
                "emails": emails,
                "digest": None,
                "error": None,
                "retry_count": 0,
            }
        )

        # -------------------------------------------------
        # 6. VALIDATION
        # -------------------------------------------------

        job_manager.update_job(
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
        # 7. COMPLETE
        # -------------------------------------------------

        job_manager.update_job(
            job_id,
            status=JobStatus.COMPLETED,
            step=JobStep.COMPLETED,
            progress=100,
            result=digest.model_dump(),
        )

        print(
            f"[EMAIL WORKER] "
            f"{job_id} completed successfully"
        )

    except Exception as exc:

        print(
            f"[EMAIL WORKER] "
            f"{job_id} failed: {exc}"
        )

        job_manager.update_job(
            job_id,
            status=JobStatus.FAILED,
            progress=100,
            error=str(exc),
        )

    finally:

        await browser.close()

        print(
            f"[EMAIL WORKER] "
            f"{job_id} browser closed"
        )