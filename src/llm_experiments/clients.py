import asyncio
from argparse import ArgumentParser, Namespace

from dotenv import load_dotenv
from langchain_community.tools.playwright.utils import create_async_playwright_browser
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.memory import BaseCheckpointSaver, MemorySaver

from llm_experiments import tools
from llm_experiments.agent import Agent
from llm_experiments.llm import create_model


def parse_args() -> Namespace:
    parser = ArgumentParser()
    opt = parser.add_argument
    opt(
        "--agent",
        "-a",
        choices=["search", "shell", "browser", "shell_w_search", "sql", "slack", "python-repl"],
        default="search",
    )
    opt("--model", "-m", type=str, default="4o-mini")
    return parser.parse_args()


async def main() -> None:
    load_dotenv()
    args = parse_args()
    config = Config(model=create_model(args.model), memory=MemorySaver())
    match args.agent:
        case "search":
            await search(config)
        case "shell":
            await shell(config)
        case "browser":
            await browser(config)
        case "shell_w_search":
            await shell_w_search(config)
        case "sql":
            await sql(config)
        case "slack":
            await slack(config)
        case "python-repl":
            await python_repl(config)
        case _:
            raise ValueError(f"unknown agent: {args.agent}")


class Config:
    def __init__(self, model: BaseChatModel, memory: BaseCheckpointSaver):
        self.model = model
        self.memory = memory
        self.configurable = {"configurable": {"thread_id": "default"}}


async def search(config: Config) -> None:
    toolkit = [tools.tavily(), tools.duckduckgo(), tools.serper(), tools.wikipedia()]
    agent = Agent(model=config.model, tools=toolkit, memory=config.memory, config=config.configurable)
    await agent.start_interactive_chat()


async def shell_w_search(config: Config) -> None:
    toolkit = [tools.shell(ask_human_input=True), tools.tavily(), tools.duckduckgo(), tools.serper()]
    agent = Agent(model=config.model, tools=toolkit, memory=config.memory, config=config.configurable)
    await agent.start_interactive_chat()


async def shell(config: Config) -> None:
    toolkit = [tools.shell(ask_human_input=True)]
    agent = Agent(model=config.model, tools=toolkit, memory=config.memory, config=config.configurable)
    await agent.start_interactive_chat()


# fixme
async def slack(config: Config) -> None:
    toolkit = [*tools.slack_tools()]
    agent = Agent(model=config.model, tools=toolkit, memory=config.memory, config=config.configurable)
    await agent.start_interactive_chat()


async def python_repl(config: Config) -> None:
    toolkit = [tools.python_repl()]
    agent = Agent(model=config.model, tools=toolkit, memory=config.memory, config=config.configurable)
    await agent.start_interactive_chat()


async def sql(config: Config) -> None:
    toolkit = [*tools.sql_tools(config.model, "sql"), tools.shell(), tools.duckduckgo()]
    agent = Agent(model=config.model, tools=toolkit, memory=config.memory, config=config.configurable)
    await agent.start_interactive_chat()


# fixme
async def browser(config: Config) -> None:
    import nest_asyncio

    nest_asyncio.apply()

    async with create_async_playwright_browser(headless=False) as async_browser:
        toolkit = await tools.browser_tools(async_browser)
        agent = Agent(model=config.model, tools=toolkit, memory=config.memory, config=config.configurable)
        await agent.start_interactive_chat()


if __name__ == "__main__":
    asyncio.run(main())
