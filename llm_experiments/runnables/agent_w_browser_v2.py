import asyncio
from argparse import ArgumentParser
from langchain.agents import initialize_agent, AgentType
from playwright.async_api import async_playwright
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from llm_experiments.models import instantiate_chat


async def run(query):
    model = instantiate_chat("4o-mini")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=browser)
        tools = toolkit.get_tools()
        agent = initialize_agent(tools, model, agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
        result = await agent.ainvoke(query)
        print(result)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--query", "-q", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    asyncio.run(run(args.query))


if __name__ == "__main__":
    main()
