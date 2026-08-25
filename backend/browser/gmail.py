import asyncio
import re

from backend.services.models import EmailMessage


class GmailService:
    INBOX_URL = "https://mail.google.com/mail/u/0/#inbox"
    OLDER_BUTTON = '[aria-label="Older"]'

    def __init__(self, page):
        self.page = page

    async def open_inbox(self):
        await self.page.goto(self.INBOX_URL)

        await self.page.wait_for_load_state("domcontentloaded")

        await self.page.locator('tr[role="row"]').first.wait_for(
            state="visible"
        )

    @staticmethod
    def clean_text(value: str) -> str:
        """
        Clean text extracted from Gmail's DOM.
        """
        # Remove invisible formatting characters.
        # value = re.sub(r"[\u200b-\u200f\u2060\ufeff]", "", value)
        value = re.sub(
            r"[\u0000-\u001F\u007F-\u009F\u200B-\u200F\u202A-\u202E\u2060\u2066-\u2069\uFEFF]",
            "",
            value,
        )

        # Normalize non-breaking spaces.
        value = value.replace("\xa0", " ")

        # Normalize whitespace.
        value = re.sub(r"\s+", " ", value)

        return value.strip()

    async def _go_to_older_page(self) -> bool:
        """
        Advance to the next page of the inbox. Returns False when
        there is no next page, so callers stop instead of looping.
        """

        older = self.page.locator(self.OLDER_BUTTON)

        if await older.count() == 0:
            return False

        if await older.first.get_attribute("aria-disabled") == "true":
            return False

        before = await self._first_thread_id()

        await older.first.click()

        # Gmail swaps the list in place, so wait for the top row to
        # change rather than for a navigation that never happens.
        for _ in range(40):
            await asyncio.sleep(0.5)

            if await self._first_thread_id() not in (None, before):
                return True

        print("[GMAIL] Next page did not load in time")

        return False

    async def _first_thread_id(self) -> str | None:
        rows = self.page.locator("[data-legacy-thread-id]")

        if await rows.count() == 0:
            return None

        return await rows.first.get_attribute(
            "data-legacy-thread-id"
        )

    async def get_recent_emails(
        self,
        limit: int = 10,
    ) -> list[EmailMessage]:
        """
        Extract up to `limit` messages, paging through the inbox
        with the Older button when one page is not enough.
        """

        emails: list[EmailMessage] = []
        seen: set[str] = set()

        while len(emails) < limit:
            batch = await self._extract_current_page(
                limit - len(emails),
                seen,
            )

            emails.extend(batch)

            if len(emails) >= limit:
                break

            if not await self._go_to_older_page():
                break

            print(
                f"[GMAIL] {len(emails)}/{limit} so far, "
                f"loading older page..."
            )

        return emails

    async def _extract_current_page(
        self,
        limit: int,
        seen: set[str],
    ) -> list[EmailMessage]:

        rows = self.page.locator('tr[role="row"]')

        row_count = await rows.count()

        emails: list[EmailMessage] = []

        for index in range(row_count):
            if len(emails) >= limit:
                break

            row = rows.nth(index)

            # count() does not auto-wait, so non-email rows are
            # skipped instantly instead of burning a 30s timeout.
            if await row.locator("span[email]").count() == 0:
                continue

            try:
                # --------------------------------------------------
                # Sender
                # --------------------------------------------------

                sender = row.locator("span[email]").first

                sender_name = await sender.get_attribute("name")
                sender_email = await sender.get_attribute("email")

                # --------------------------------------------------
                # Subject
                # --------------------------------------------------

                subject_element = row.locator(
                    '[data-thread-id][data-legacy-thread-id]'
                ).first

                subject = self.clean_text(
                    await subject_element.inner_text()
                )

                # --------------------------------------------------
                # Snippet
                # --------------------------------------------------

                snippet_element = row.locator(".y2").first

                snippet = self.clean_text(
                    await snippet_element.inner_text()
                )

                # Gmail puts a visual "-" separator before snippets.
                if snippet.startswith("-"):
                    snippet = snippet[1:].strip()

                # --------------------------------------------------
                # Timestamp
                # --------------------------------------------------

                timestamp_element = row.locator(
                    'span[aria-label][title]'
                ).first

                timestamp = await timestamp_element.get_attribute("title")

                # --------------------------------------------------
                # Thread ID
                # --------------------------------------------------

                thread_id = await subject_element.get_attribute(
                    "data-thread-id"
                )

                # Gmail resolves #inbox/<legacy-id> to the thread.
                legacy_id = await subject_element.get_attribute(
                    "data-legacy-thread-id"
                )

                # Pages can overlap when mail arrives mid-scrape.
                key = legacy_id or thread_id

                if key:
                    if key in seen:
                        continue

                    seen.add(key)

                link = (
                    f"{self.INBOX_URL}/{legacy_id}"
                    if legacy_id
                    else None
                )

                emails.append(
                    EmailMessage(
                        sender_name=sender_name or "",
                        sender_email=sender_email or "",
                        subject=subject,
                        snippet=snippet,
                        timestamp=timestamp or "",
                        thread_id=thread_id,
                        link=link,
                    )
                )

            except Exception as exc:
                # Gmail's row list also contains ads and section
                # headers. Skip anything that doesn't parse.
                print(
                    f"[GMAIL] Skipped row {index}: {exc}"
                )

                continue

        return emails