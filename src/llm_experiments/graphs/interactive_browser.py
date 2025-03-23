import asyncio

from playwright.async_api import async_playwright

from llm_experiments.agents import BrowserAgent
from llm_experiments.llm import create_model


async def main():
    async with async_playwright() as p:
        model = create_model()
        browser_agent = BrowserAgent(model, p)
        agent = await browser_agent.agent(verbose=True)
        while True:
            user_input = input("user: ")
            if user_input == "q":
                break
            res = await agent.ainvoke({"input": user_input})
            print("assistant:", res["output"])


if __name__ == "__main__":
    asyncio.run(main())
