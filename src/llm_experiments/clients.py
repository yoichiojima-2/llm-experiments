import asyncio
from argparse import ArgumentParser

from langgraph.checkpoint.memory import MemorySaver

from llm_experiments import tools
from llm_experiments.agent import Agent
from llm_experiments.llm import create_model


async def search(model, memory, verbose, config):
    agent = Agent(
        model=create_model(model),
        memory=memory,
        tools=[tools.tavily(), tools.duckduckgo(), tools.serper(), tools.wikipedia()],
        verbose=verbose,
        config=config,
    )
    await agent.interactive_chat()


async def shell(model, memory, verbose, config):
    agent = Agent(
        model=create_model(model),
        memory=memory,
        tools=[tools.shell(ask_human_input=True)],
        verbose=verbose,
        config=config,
    )
    await agent.interactive_chat()


def parse_args():
    parser = ArgumentParser()
    opt = parser.add_argument
    opt("--agent", "-a", choices=["search", "shell"])
    opt("--model", "-m", type=str, default="4o-mini")
    opt("--verbose", "-v", action="store_true", default=False)
    return parser.parse_args()


async def main():
    config = {"configurable": {"thread_id": "search"}}
    args = parse_args()
    memory = MemorySaver()
    match args.agent:
        case "search":
            await search(model=args.model, memory=memory, verbose=args.verbose, config=config)
        case "shell":
            await shell(model=args.model, memory=memory, verbose=args.verbose, config=config)
        case _:
            raise ValueError(f"Unknown agent: {args.agent}")


asyncio.run(main())
