import asyncio
from pathlib import Path

from camoufox.async_api import AsyncCamoufox


# Firefox locks user_data_dir, so two launches against the same
# profile fail. Serialize them instead.
_profile_locks: dict[str, asyncio.Lock] = {}


class BrowserManager:

    def __init__(
        self,
        user_id: str,
        profile_name: str = "google",
    ):
        self.user_id = user_id

        self.profile_dir = (
            Path("browser_profiles")
            / user_id
            / profile_name
        )

        self.camoufox = None
        self.context = None
        self._lock = None

    async def launch(self, headless: bool):
        lock = _profile_locks.setdefault(
            str(self.profile_dir),
            asyncio.Lock(),
        )

        await lock.acquire()

        # Only claim it once acquire() has actually returned. If the
        # wait is cancelled, close() must not release someone else's
        # lock.
        self._lock = lock

        self.profile_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        print(
            f"[BROWSER] User={self.user_id}"
        )

        print(
            f"[BROWSER] Profile={self.profile_dir}"
        )

        self.camoufox = AsyncCamoufox(
            headless=headless,
            persistent_context=True,
            user_data_dir=str(
                self.profile_dir
            ),
        )

        self.context = (
            await self.camoufox.__aenter__()
        )

        print("[BROWSER] Browser launched")

        if self.context.pages:
            page = self.context.pages[0]
        else:
            page = await self.context.new_page()

        return page

    async def close(self):
        try:
            if self.camoufox is not None:
                print("[BROWSER] Closing browser...")

                await self.camoufox.__aexit__(
                    None,
                    None,
                    None,
                )

                print("[BROWSER] Browser closed")

        except Exception as exc:
            # close() usually runs in a finally block. Raising
            # here would replace whatever actually went wrong.
            print(f"[BROWSER] Close failed: {exc}")

        finally:
            self.camoufox = None
            self.context = None

            # Release even if __aexit__ blew up, or the
            # profile stays locked forever.
            if self._lock is not None:
                self._lock.release()
                self._lock = None
