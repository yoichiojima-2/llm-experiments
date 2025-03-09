import asyncio
from argparse import ArgumentParser

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langgraph.graph import START, MessagesState, StateGraph
from playwright.async_api import async_playwright

from nodes import SpotifyNode, SupervisorNode

from langgraph.checkpoint.memory import MemorySaver

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


async def run(query, thread_id="1"):
    model = init_chat_model("gpt-4o-mini", model_provider="openai")
    config = {"configurable": {"thread_id": thread_id}}

    memory = MemorySaver()
    supervisor = SupervisorNode(model).node()
    spotify = SpotifyNode(model, checkpointer=memory).node()

    graph = StateGraph(MessagesState)
    graph.add_node("supervisor", supervisor)
    graph.add_node("spotify", spotify)
    graph.add_edge(START, "supervisor")
    app = graph.compile()
    async for i in app.astream({"messages": [query]}, config):
        print(i)




async def main():
    args = parse_args()
    await run(args.query)


if __name__ == "__main__":
    asyncio.run(main())
