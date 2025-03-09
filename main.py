import asyncio
from argparse import ArgumentParser

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langgraph.checkpoint.memory import MemorySaver
from playwright.async_api import async_playwright

import agents

load_dotenv()


def parse_args():
    parser = ArgumentParser()
    opt = parser.add_argument
    opt("-q", "--query", required=True)
    return parser.parse_args()


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


async def print_stream(stream):
    async for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()


async def run(query, thread_id="1"):
    inputs = {"messages": [("user", query)]}
    config = {"configurable": {"thread_id": thread_id}}

    model = init_chat_model("gpt-4o-mini", model_provider="openai")

    async with Playwright() as _:
        memory = MemorySaver()
        agent = agents.FileAgent().agent(model, checkpointer=memory)

        res = agent.astream(inputs, config, stream_mode="values")
        await print_stream(res)


async def main():
    args = parse_args()
    await run(args.query)


if __name__ == "__main__":
    asyncio.run(main())
