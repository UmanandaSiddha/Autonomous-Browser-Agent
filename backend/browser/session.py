from playwright.async_api import Page

from backend.browser.manager import BrowserManager


class BrowserSession:
    def __init__(self):
        self.manager = BrowserManager()
        self.page: Page | None = None

    async def start(self):
        self.page = await self.manager.launch()
        return self.page

    async def close(self):
        await self.manager.close()
        self.page = None