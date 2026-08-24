# from pathlib import Path

# from camoufox.async_api import AsyncCamoufox


# class BrowserManager:
#     def __init__(self, profile_dir: str = "browser_profiles/google"):
#         self.profile_dir = Path(profile_dir)
#         self.camoufox = None
#         self.context = None

#     async def launch(self):
#         self.profile_dir.mkdir(parents=True, exist_ok=True)

#         self.camoufox = AsyncCamoufox(
#             headless=True,
#             persistent_context=True,
#             user_data_dir=str(self.profile_dir),
#         )

#         self.context = await self.camoufox.__aenter__()

#         if self.context.pages:
#             page = self.context.pages[0]
#         else:
#             page = await self.context.new_page()

#         return page

#     async def close(self):
#         if self.camoufox is not None:
#             await self.camoufox.__aexit__(None, None, None)

#         self.camoufox = None
#         self.context = None


from pathlib import Path

from camoufox.async_api import AsyncCamoufox


class BrowserManager:
    def __init__(
        self,
        profile_dir: str = "browser_profiles/google",
    ):
        self.profile_dir = Path(profile_dir)
        self.camoufox = None
        self.context = None

    async def launch(self, headless: bool = True):
        """
        Launch a persistent Camoufox browser.

        headless=True:
            Used for normal background automation.

        headless=False:
            Used for interactive authentication/setup.
        """

        self.profile_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        print(
            f"[BROWSER] Launching Camoufox "
            f"(headless={headless})"
        )

        self.camoufox = AsyncCamoufox(
            headless=headless,
            persistent_context=True,
            user_data_dir=str(self.profile_dir),
        )

        self.context = await self.camoufox.__aenter__()

        if self.context.pages:
            page = self.context.pages[0]
        else:
            page = await self.context.new_page()

        print("[BROWSER] Browser launched")

        return page

    async def close(self):
        """Close the browser and its Playwright context."""

        if self.camoufox is not None:
            print("[BROWSER] Closing browser...")

            await self.camoufox.__aexit__(
                None,
                None,
                None,
            )

        self.camoufox = None
        self.context = None

        print("[BROWSER] Browser closed")