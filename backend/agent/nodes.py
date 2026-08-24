from backend.browser.auth import GmailAuth
from backend.browser.gmail import GmailService
from backend.browser.manager import BrowserManager
from backend.llm.ollama import OllamaService

from .state import AgentState


# async def check_auth(state: AgentState) -> dict:
#     print("[AUTH] Starting authentication check")

#     browser = BrowserManager()

#     try:
#         page = await browser.launch()

#         auth = GmailAuth()

#         authenticated = await auth.is_authenticated(page)

#         print(f"[AUTH] Authenticated = {authenticated}")

#         if not authenticated:
#             return {
#                 "authenticated": False,
#                 "error": (
#                     "Gmail is not authenticated. "
#                     "Please log in using the browser profile."
#                 ),
#             }

#         return {
#             "authenticated": True,
#             "error": None,
#         }

#     except Exception as exc:
#         return {
#             "authenticated": False,
#             "error": str(exc),
#         }

#     finally:
#         await browser.close()

async def check_auth(state: AgentState, page) -> dict:
    print("[AUTH] Starting authentication check")

    try:
        auth = GmailAuth()

        authenticated = await auth.is_authenticated(page)

        print(f"[AUTH] Authenticated = {authenticated}")

        if not authenticated:
            return {
                "authenticated": False,
                "error": (
                    "Gmail is not authenticated. "
                    "Please log in using the browser profile."
                ),
            }

        return {
            "authenticated": True,
            "error": None,
        }

    except Exception as exc:
        print(f"[AUTH] Error: {exc}")

        return {
            "authenticated": False,
            "error": str(exc),
        }


async def extract_emails(state: AgentState, page) -> dict:
    print("[EXTRACT] Starting email extraction")

    try:
        gmail = GmailService(page)

        await gmail.open_inbox()

        emails = await gmail.get_recent_emails(limit=10)

        print(f"[EXTRACT] Extracted {len(emails)} emails")

        return {
            "emails": emails,
            "error": None,
        }

    except Exception as exc:
        print(f"[EXTRACT] Error: {exc}")

        return {
            "error": str(exc),
        }


async def summarize_emails(state: AgentState) -> dict:
    print("[LLM] Starting summarization")

    print(
        f"[LLM] Emails received: {len(state['emails'])}"
    )

    if not state["emails"]:
        print("[LLM] No emails found")

        return {
            "error": "No emails available for summarization.",
        }

    llm = OllamaService()

    try:
        digest = await llm.summarize_emails(
            state["emails"]
        )

        print(type(digest))
        print(digest)

        print("[LLM] Summarization complete")

        print(
            f"[LLM] Digest exists: {digest is not None}"
        )

        return {
            "digest": digest,
            "error": None,
        }

    except Exception as exc:
        print(f"[LLM] Error: {exc}")

        return {
            "error": str(exc),
        }


async def validate_digest(state: AgentState) -> dict:
    print("[VALIDATE] Running validation")

    digest = state.get("digest")

    print(
        f"[VALIDATE] Digest exists = {digest is not None}"
    )

    if digest is None:
        print("[VALIDATE] Digest is None")

        return {
            "error": "No digest was produced.",
        }

    print(
        f"[VALIDATE] Summary length = {len(digest.summary)}"
    )

    if not digest.summary.strip():
        return {
            "error": "Digest summary is empty.",
        }

    return {
        "error": None,
    }


async def prepare_retry(state: AgentState) -> dict:
    return {
        "retry_count": state["retry_count"] + 1,
        "error": None,
    }


async def handle_error(state: AgentState) -> dict:
    return {
        "error": state.get("error") or "Unknown agent error",
    }