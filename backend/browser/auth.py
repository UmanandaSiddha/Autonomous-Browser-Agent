import asyncio
import time

from playwright.async_api import Page


class GmailAuth:
    GMAIL_URL = "https://mail.google.com/mail/u/0/#inbox"

    # Google only issues these once authentication completes, so
    # they flip well before Gmail has finished rendering.
    SESSION_COOKIES = ("SID", "__Secure-1PSID")

    async def _has_session_cookie(self, page: Page) -> bool:
        names = {
            cookie["name"]
            for cookie in await page.context.cookies()
        }

        return any(
            name in names
            for name in self.SESSION_COOKIES
        )

    async def _confirm_inbox(self, context) -> bool:
        """
        Google has issued a session -- now prove Gmail actually
        opens. Done on a page we control: the user may have
        logged in from a different tab, or never left the
        marketing page they landed on.
        """

        page = await context.new_page()

        try:
            await page.goto(self.GMAIL_URL)
            await page.wait_for_load_state(
                "domcontentloaded"
            )

            return await self._check_inbox(page)

        except Exception as exc:
            print(
                f"[AUTH] Inbox check failed: {exc}"
            )

            return False

        finally:
            try:
                await page.close()
            except Exception:
                pass

    async def _check_inbox(self, page: Page) -> bool:
        """
        Check whether Gmail's inbox is currently visible.
        Does not navigate.
        """

        email_rows = page.locator(
            'tr[role="row"]'
        )

        try:
            await email_rows.first.wait_for(
                state="visible",
                timeout=10_000,
            )

            return await email_rows.count() > 0

        except Exception:
            return False

    async def is_authenticated(
        self,
        page: Page,
    ) -> bool:
        """
        Check whether the current persistent browser
        profile is authenticated with Gmail.
        """

        print(
            "[AUTH] Checking Gmail authentication..."
        )

        await page.goto(self.GMAIL_URL)
        await page.wait_for_load_state(
            "domcontentloaded"
        )

        authenticated = await self._check_inbox(
            page
        )

        print(
            f"[AUTH] Gmail authenticated = "
            f"{authenticated}"
        )

        return authenticated

    async def wait_for_authentication(
        self,
        page: Page,
        timeout: int = 300,
        poll_interval: float = 2.0,
    ) -> bool:
        """
        Open Gmail in an interactive browser and wait
        for the user to complete Google authentication.

        Returns as soon as Google has issued a session and the
        inbox has rendered, so the browser closes immediately
        instead of making the user sit through the timeout.
        """

        print(
            "[AUTH] Starting interactive "
            "Gmail authentication"
        )

        await page.goto(self.GMAIL_URL)
        await page.wait_for_load_state(
            "domcontentloaded"
        )

        print(
            "[AUTH] Please complete Google login "
            "in the browser window..."
        )

        deadline = time.monotonic() + timeout
        had_session = None

        while time.monotonic() < deadline:

            if page.is_closed():
                print(
                    "[AUTH] Browser was closed before "
                    "login completed"
                )

                return False

            try:
                has_session = await self._has_session_cookie(page)

            except Exception as exc:
                # Almost always the user closing the window.
                print(
                    f"[AUTH] Browser is no longer "
                    f"available: {exc}"
                )

                return False

            if has_session != had_session:
                print(
                    f"[AUTH] session={has_session} "
                    f"url={page.url[:80]}"
                )

                had_session = has_session

            if has_session:
                # The session can appear mid-flow (before a
                # security check finishes), so keep re-checking
                # until Gmail itself opens.
                if await self._confirm_inbox(page.context):
                    elapsed = int(
                        timeout - (deadline - time.monotonic())
                    )

                    print(
                        f"[AUTH] Gmail login confirmed "
                        f"after {elapsed}s"
                    )

                    return True

            await asyncio.sleep(poll_interval)

        print(
            "[AUTH] Gmail authentication timed out "
            f"after {timeout} seconds"
        )

        return False
