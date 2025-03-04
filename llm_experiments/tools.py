import sys
from pathlib import Path

sys.path.append(str(Path().cwd().parent))

from playwright.async_api import async_playwright
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit


class BrowserManager:
    def __init__(self):
        self.playwright = None
        self.browser = None
        self.tools = None

    async def start(self):
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=False)
        self.tools = PlayWrightBrowserToolkit.from_browser(async_browser=self.browser).get_tools()

    async def stop(self):
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()

    def get_tools(self):
        if self.tools is None:
            raise RuntimeError("BrowserManager is not started. Call `await start()` first.")
        return self.tools

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()