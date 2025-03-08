import asyncio

from langchain.chat_models import init_chat_model
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from playwright.async_api import async_playwright


class Browser:
    async def __aenter__(self):
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=False)
        toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=self.browser)
        self.tools = toolkit.get_tools()
        return self

    async def __aexit__(self, *a):
        await self.browser.close()
        await self.playwright.stop()


async def main():
    async with Browser() as b:
        llm = init_chat_model("gpt-4o-mini", model_provider="openai")
        llm_w_tools = llm.bind_tools(b.tools)
        res = await llm_w_tools.ainvoke("search google")


if __name__ == "__main__":
    asyncio.run(main())
