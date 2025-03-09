import asyncio

from langchain.chat_models import init_chat_model
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langgraph.prebuilt import create_react_agent
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


async def print_stream(stream):
    async for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()


async def main():
    async with Browser() as b:
        model = init_chat_model("gpt-4o-mini", model_provider="openai")
        graph = create_react_agent(model, tools=b.tools)
        inputs = {"messages": [("user", "what is the weather in tokyo right now?")]}
        await print_stream(graph.astream(inputs, stream_mode="values"))


if __name__ == "__main__":
    asyncio.run(main())
