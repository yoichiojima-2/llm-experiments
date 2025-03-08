import sys
from pathlib import Path

sys.path.append(str(Path().cwd().parent))

from playwright.async_api import async_playwright
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit


class Browser:
    async def __aenter__(self):
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=False)
        self.tools = PlayWrightBrowserToolkit.from_browser(async_browser=self.browser).get_tools()
        return self

    async def __aexit__(self, *a):
        await self.browser.close()
        await self.playwright.stop()
