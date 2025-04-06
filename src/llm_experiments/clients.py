import asyncio
from argparse import ArgumentParser

from dotenv import load_dotenv
from langchain.agents import AgentType, initialize_agent
from langchain_community.tools.playwright.utils import create_async_playwright_browser
from langgraph.checkpoint.memory import MemorySaver

from llm_experiments import tools
from llm_experiments.agent import Agent, create_executor
from llm_experiments.llm import create_model


async def search(model, memory, verbose, config):
    model = create_model(model)
    toolkit = [tools.tavily(), tools.duckduckgo(), tools.serper()]
    executor = create_executor(model, toolkit, verbose)
    agent = Agent(
        executor=executor,
        model=model,
        memory=memory,
        tools=toolkit,
        verbose=verbose,
        config=config,
    )
    await agent.interactive_chat()


async def shell_w_web(model, memory, verbose, config):
    model = create_model(model)
    toolkit = [tools.shell(ask_human_input=True), tools.tavily(), tools.duckduckgo(), tools.serper()]
    executor = create_executor(model, toolkit, verbose)
    agent = Agent(
        executor=executor,
        model=model,
        memory=memory,
        tools=toolkit,
        verbose=verbose,
        config=config,
    )
    await agent.interactive_chat()


async def shell(model, memory, verbose, config):
    model = create_model(model)
    toolkit = [tools.shell(ask_human_input=True)]
    executor = create_executor(model, toolkit, verbose)
    agent = Agent(
        executor=executor,
        model=model,
        memory=memory,
        tools=toolkit,
        verbose=verbose,
        config=config,
    )
    await agent.interactive_chat()


# fixme
async def browser(model, memory, verbose, config):
    import nest_asyncio

    nest_asyncio.apply()

    async with create_async_playwright_browser(headless=False) as async_browser:
        model = create_model(model)
        toolkit = await tools.browser_tools(async_browser)
        executor = initialize_agent(
            toolkit,
            model,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
        )
        agent = Agent(
            executor=executor,
            model=model,
            memory=memory,
            tools=toolkit,
            verbose=verbose,
            config=config,
        )
        await agent.interactive_chat()


def parse_args():
    parser = ArgumentParser()
    opt = parser.add_argument
    opt("--agent", "-a", choices=["search", "shell", "browser", "shell_w_web"], default="search")
    opt("--model", "-m", type=str, default="4o-mini")
    opt("--verbose", "-v", action="store_true", default=False)
    return parser.parse_args()


async def main():
    load_dotenv()
    config = {"configurable": {"thread_id": "search"}}
    args = parse_args()
    memory = MemorySaver()
    match args.agent:
        case "search":
            await search(model=args.model, memory=memory, verbose=args.verbose, config=config)
        case "shell":
            await shell(model=args.model, memory=memory, verbose=args.verbose, config=config)
        case "browser":
            await browser(model=args.model, memory=memory, verbose=args.verbose, config=config)
        case "shell_w_web":
            await shell_w_web(model=args.model, memory=memory, verbose=args.verbose, config=config)
        case _:
            raise ValueError(f"Unknown agent: {args.agent}")


asyncio.run(main())
