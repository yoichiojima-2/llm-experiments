import asyncio
from argparse import ArgumentParser

from langchain_community.tools.playwright.utils import create_async_playwright_browser
from langgraph.checkpoint.memory import MemorySaver

from llm_experiments import tools as t
from llm_experiments.agent import Agent
from llm_experiments.llm import create_model


def parse_args():
    parser = ArgumentParser()
    opt = parser.add_argument
    opt(
        "--agent",
        "-a",
        choices=["search", "shell", "browser", "shell_w_search", "sql", "slack", "python-repl", "browser_w_search"],
        default="search",
    )
    opt("--model", "-m", type=str, choices=["4o-mini", "o3-mini", "deepseek", "llama", "gemini"], default="4o-mini")
    return parser.parse_args()


async def main():
    args = parse_args()
    model = create_model(args.model)
    memory = MemorySaver()
    config = {"configurable": {"thread_id": "default"}}
    match args.agent:
        case "search":
            await search(model, memory, config)
        case "shell":
            await shell(model, memory, config)
        case "browser":
            await browser(model, memory, config)
        case "shell_w_search":
            await shell_w_search(model, memory, config)
        case "sql":
            await sql(model, memory, config)
        case "slack":
            await slack(model, memory, config)
        case "python-repl":
            await python_repl(model, memory, config)
        case "browser_w_search":
            await browser_w_search(model, memory, config)
        case _:
            raise ValueError(f"unknown agent: {args.agent}")


async def search(model, memory, config):
    agent = Agent(
        model=model,
        tools=[t.tavily(), t.duckduckgo(), t.serper(), t.wikipedia()],
        memory=memory,
        config=config,
    )
    await agent.start_interactive_chat()


async def shell_w_search(model, memory, config):
    agent = Agent(
        model=model,
        tools=[t.shell(ask_human_input=True), t.tavily(), t.duckduckgo(), t.serper()],
        memory=memory,
        config=config,
    )
    await agent.start_interactive_chat()


async def shell(model, memory, config):
    agent = Agent(
        model=model,
        tools=[t.shell(ask_human_input=True)],
        memory=memory,
        config=config,
    )
    await agent.start_interactive_chat()


async def slack(model, memory, config):
    agent = Agent(
        model=model,
        tools=t.slack_tools(),
        memory=memory,
        config=config,
    )
    await agent.start_interactive_chat()


async def python_repl(model, memory, config):
    agent = Agent(
        model=model,
        tools=[t.python_repl()],
        memory=memory,
        config=config,
    )
    await agent.start_interactive_chat()


async def sql(model, memory, config):
    agent = Agent(
        model=model,
        tools=[*t.sql_tools(model, "sql"), t.shell(), t.duckduckgo()],
        memory=memory,
        config=config,
    )
    await agent.start_interactive_chat()


async def browser(model, memory, config):
    import nest_asyncio

    nest_asyncio.apply()
    async with create_async_playwright_browser(headless=False) as async_browser:
        toolkit = await t.browser_tools(async_browser)
        agent = Agent(
            model=model,
            tools=toolkit,
            memory=memory,
            config=config,
        )
        await agent.start_interactive_chat()


async def browser_w_search(model, memory, config):
    import nest_asyncio

    nest_asyncio.apply()
    async with create_async_playwright_browser(headless=False) as async_browser:
        browser_tools = await t.browser_tools(async_browser)
        agent = Agent(
            model=model,
            tools=[*browser_tools, t.duckduckgo(), t.serper(), t.wikipedia(), t.tavily()],
            memory=memory,
            config=config,
        )
        await agent.start_interactive_chat()


if __name__ == "__main__":
    asyncio.run(main())
