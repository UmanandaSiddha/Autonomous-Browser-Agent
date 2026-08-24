from pathlib import Path

from camoufox.async_api import AsyncCamoufox


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

    async def launch(self, headless: bool):
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
            headless,
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
        if self.camoufox is not None:
            print("[BROWSER] Closing browser...")

            await self.camoufox.__aexit__(
                None,
                None,
                None,
            )

            print("[BROWSER] Browser closed")

        self.camoufox = None
        self.context = None