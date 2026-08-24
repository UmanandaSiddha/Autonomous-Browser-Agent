from backend.llm.ollama import OllamaService

from .state import AgentState


async def summarize_emails(
    state: AgentState,
) -> dict:

    print("[LLM] Starting summarization")

    emails = state.get("emails", [])

    print(
        f"[LLM] Emails received: {len(emails)}"
    )

    if not emails:
        return {
            "error": "No emails available for summarization."
        }

    llm = OllamaService()

    try:

        digest = await llm.summarize_emails(
            emails
        )

        print(
            "[LLM] Summarization complete"
        )

        print(
            f"[LLM] Digest exists: "
            f"{digest is not None}"
        )

        return {
            "digest": digest,
            "error": None,
        }

    except Exception as exc:

        print(
            f"[LLM] Error: {exc}"
        )

        return {
            "digest": None,
            "error": str(exc),
        }


async def validate_digest(
    state: AgentState,
) -> dict:

    print("[VALIDATE] Running validation")

    digest = state.get("digest")

    print(
        f"[VALIDATE] Digest exists = "
        f"{digest is not None}"
    )

    if digest is None:

        # Keep the summarizer's error if there is one, otherwise
        # the real cause is replaced by a generic message.
        return {
            "error": (
                state.get("error")
                or "No digest was produced."
            )
        }

    if not digest.summary.strip():

        return {
            "error": "Digest summary is empty."
        }

    print(
        f"[VALIDATE] Summary length = "
        f"{len(digest.summary)}"
    )

    return {
        "error": None
    }


def route_after_validation(
    state: AgentState,
):

    error = state.get("error")

    retry_count = state.get(
        "retry_count",
        0,
    )

    print(
        f"[ROUTER] Validation error = "
        f"{error}"
    )

    print(
        f"[ROUTER] Retry count = "
        f"{retry_count}"
    )

    if error is None:
        return "success"

    if retry_count < 2:
        return "retry"

    return "error"


async def prepare_retry(
    state: AgentState,
) -> dict:

    retry_count = state.get(
        "retry_count",
        0,
    )

    print(
        f"[RETRY] Preparing retry "
        f"{retry_count + 1}"
    )

    return {
        "retry_count": retry_count + 1,
        "error": None,
        "digest": None,
    }


async def handle_error(
    state: AgentState,
) -> dict:

    error = state.get(
        "error"
    ) or "Unknown agent error"

    print(
        f"[ERROR] Agent failed: {error}"
    )

    return {
        "error": error
    }