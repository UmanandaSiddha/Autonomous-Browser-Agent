from fastapi import APIRouter

from backend.browser.auth import GmailAuth
from backend.browser.manager import BrowserManager

from backend.api.schemas import (
    AuthConnectResponse,
    AuthStatusResponse,
)

router = APIRouter(
    prefix="/api/auth",
    tags=["Authentication"],
)

@router.get(
    "/gmail/status",
    response_model=AuthStatusResponse,
)
async def gmail_auth_status():
    """
    Check whether the persistent Gmail browser
    profile is currently authenticated.
    """

    print("[API][AUTH] Checking Gmail authentication")

    browser = BrowserManager()

    try:
        page = await browser.launch(
            headless=True
        )

        auth = GmailAuth()

        authenticated = await auth.is_authenticated(
            page
        )

        print(
            f"[API][AUTH] Authenticated = "
            f"{authenticated}"
        )

        return AuthStatusResponse(
            authenticated=authenticated
        )

    except Exception as exc:
        print(
            f"[API][AUTH] Error: {exc}"
        )

        return AuthStatusResponse(
            authenticated=False
        )

    finally:
        await browser.close()


@router.post(
    "/gmail/connect",
    response_model=AuthConnectResponse,
)
async def connect_gmail():
    """
    Launch an interactive browser so the user
    can authenticate their Google account.

    The persistent browser profile stores the
    resulting authentication state.
    """

    print(
        "[API][AUTH] Starting Gmail connection"
    )

    browser = BrowserManager()

    try:
        page = await browser.launch(
            headless=False
        )

        auth = GmailAuth()

        print(
            "[API][AUTH] Waiting for Gmail login..."
        )

        authenticated = await auth.wait_for_authentication(
            page
        )

        if authenticated:
            print(
                "[API][AUTH] Gmail authentication "
                "successful"
            )

            return AuthConnectResponse(
                authenticated=True,
                message="Gmail connected successfully.",
            )

        print(
            "[API][AUTH] Gmail authentication "
            "failed or timed out"
        )

        return AuthConnectResponse(
            authenticated=False,
            message=(
                "Gmail authentication was not completed."
            ),
        )

    except Exception as exc:
        print(
            f"[API][AUTH] Connection error: {exc}"
        )

        return AuthConnectResponse(
            authenticated=False,
            message=str(exc),
        )

    finally:
        await browser.close()