"""
cli entrypoint
"""

import asyncio
from argparse import ArgumentParser

import nest_asyncio
from langchain_community.tools.playwright.utils import create_async_playwright_browser
from langgraph.checkpoint.memory import MemorySaver

from llm_experiments import prebuilt
from llm_experiments.llm import create_model

nest_asyncio.apply()


def parse_args():
    parser = ArgumentParser()
    opt = parser.add_argument
    opt(
        "--agent",
        "-a",
        choices=["search", "shell", "browser", "shell_w_search", "sql", "slack", "python-repl", "browser_w_search", "swe"],
        default="search",
    )
    # i've never seen deepseek works with tools
    opt("--model", "-m", type=str, choices=["4o-mini", "o3-mini", "deepseek", "llama", "gemini"], default="4o-mini")
    return parser.parse_args()


async def main():
    args = parse_args()
    model = create_model(args.model)
    memory = MemorySaver()
    config = {"configurable": {"thread_id": "default"}}
    match args.agent:
        case "swe":
            agent = await prebuilt.swe(model, memory, config, workdir="output/swe")
            await agent.interactive_chat()
        case "search":
            agent = await prebuilt.search(model, memory, config)
            await agent.interactive_chat()
        case "shell":
            agent = await prebuilt.shell(model, memory, config)
            await agent.interactive_chat()
        case "shell_w_search":
            agent = await prebuilt.shell_w_search(model, memory, config)
            await agent.interactive_chat()
        case "sql":
            agent = await prebuilt.sql(model, memory, config)
            await agent.interactive_chat()
        case "slack":
            agent = await prebuilt.slack(model, memory, config)
            await agent.interactive_chat()
        case "python-repl":
            agent = await prebuilt.python_repl(model, memory, config)
            await agent.interactive_chat()
        case "browser":
            async with create_async_playwright_browser() as b:
                agent = await prebuilt.browser(model, memory, config, b)
                await agent.interactive_chat()
        case "browser_w_search":
            async with create_async_playwright_browser() as b:
                agent = await prebuilt.browser_w_search(model, memory, config, b)
                await agent.interactive_chat()
        case _:
            raise ValueError(f"unknown agent: {args.agent}")


if __name__ == "__main__":
    asyncio.run(main())
