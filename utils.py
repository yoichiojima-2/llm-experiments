import json

from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from playwright.async_api import async_playwright


class Playwright:
    async def __aenter__(self):
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=False)
        toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=self.browser)
        self.tools = toolkit.get_tools()
        return self

    async def __aexit__(self, *a):
        await self.browser.close()
        await self.playwright.stop()


def get_last_message(messages):
    return messages["messages"][-1]


async def print_stream(stream):
    async for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()


def parse_base_model(base_model):
    return json.dumps(base_model.model_json_schema()["properties"], indent=2)
