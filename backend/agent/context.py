from backend.browser.manager import BrowserManager


class AgentContext:
    def __init__(self):
        self.browser = BrowserManager()

    async def start(self):
        print("[BROWSER] Starting browser session...")
        page = await self.browser.launch()
        print("[BROWSER] Browser session started")

        return page

    async def close(self):
        print("[BROWSER] Closing browser session...")
        await self.browser.close()
        print("[BROWSER] Browser session closed")