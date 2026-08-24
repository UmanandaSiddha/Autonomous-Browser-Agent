from playwright.async_api import Page

class GmailAuth:
    GMAIL_URL = "https://mail.google.com/mail/u/0/#inbox"

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
    ) -> bool:
        """
        Open Gmail in an interactive browser and wait
        for the user to complete Google authentication.
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

        # The profile may already be authenticated.
        if await self._check_inbox(page):
            print(
                "[AUTH] Gmail was already authenticated"
            )
            return True

        print(
            "[AUTH] Waiting for Gmail login..."
        )

        email_rows = page.locator(
            'tr[role="row"]'
        )

        try:
            await email_rows.first.wait_for(
                state="visible",
                timeout=timeout * 1000,
            )

            authenticated = (
                await email_rows.count() > 0
            )

            if authenticated:
                print(
                    "[AUTH] Gmail login completed"
                )

            return authenticated

        except Exception:
            print(
                "[AUTH] Gmail authentication "
                f"timed out after {timeout} seconds"
            )

            return False