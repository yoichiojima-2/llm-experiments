import asyncio

from langchain.chat_models import init_chat_model
from playwright.async_api import async_playwright
from llm_experiments.agents import BrowserAgent


async def main():
    async with async_playwright() as p:
        browser_agent = BrowserAgent(init_chat_model("gpt-4o-mini", model_provider="openai"), p)
        agent = await browser_agent.agent(verbose=True)
        while True:
            user_input = input("user: ")
            if user_input == "q":
                break
            res = await agent.ainvoke({"input": user_input})
            print("assistant:", res["output"])


asyncio.run(main())