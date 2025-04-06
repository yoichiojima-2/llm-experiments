import asyncio
from argparse import ArgumentParser, Namespace

from dotenv import load_dotenv
from langchain_community.tools.playwright.utils import create_async_playwright_browser
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.memory import BaseCheckpointSaver, MemorySaver

from llm_experiments import tools
from llm_experiments.agent import Agent, create_executor, interactive_chat
from llm_experiments.llm import create_model


class Config:
    def __init__(
        self, model: BaseChatModel, memory: BaseCheckpointSaver, verbose: bool, configurable: dict[str, dict[str, str]]
    ):
        self.model = model
        self.memory = memory
        self.configurable = configurable
        self.verbose = verbose


async def interactive_chat_from_tools(toolkit, config: Config):
    executor = create_executor(config.model, toolkit, config.verbose)
    agent = Agent(
        executor=executor,
        model=config.model,
        memory=config.memory,
        tools=toolkit,
        verbose=config.verbose,
        config=config.configurable,
    )
    await interactive_chat(agent)


async def search(config: Config) -> None:
    toolkit = [tools.tavily(), tools.duckduckgo(), tools.serper(), tools.wikipedia()]
    await interactive_chat_from_tools(toolkit, config)


async def shell_w_search(config: Config) -> None:
    toolkit = [tools.shell(ask_human_input=True), tools.tavily(), tools.duckduckgo(), tools.serper()]
    await interactive_chat_from_tools(toolkit, config)


async def shell(config: Config) -> None:
    toolkit = [tools.shell(ask_human_input=True)]
    await interactive_chat_from_tools(toolkit, config)


# fixme
async def browser(config: Config) -> None:
    import nest_asyncio

    nest_asyncio.apply()

    async with create_async_playwright_browser(headless=False) as async_browser:
        toolkit = await tools.browser_tools(async_browser)
        await interactive_chat_from_tools(toolkit, config)


async def sql(config: Config) -> None:
    toolkit = [*tools.sql_tools(config.model, "sql"), tools.shell(), tools.duckduckgo()]
    await interactive_chat_from_tools(toolkit, config)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    opt = parser.add_argument
    opt("--agent", "-a", choices=["search", "shell", "browser", "shell_w_search", "sql"], default="search")
    opt("--model", "-m", type=str, default="4o-mini")
    opt("--verbose", "-v", action="store_true", default=False)
    return parser.parse_args()


async def main() -> None:
    load_dotenv()
    args = parse_args()
    config = Config(
        model=create_model(args.model),
        memory=MemorySaver(),
        verbose=args.verbose,
        configurable={"configurable": {"thread_id": "search"}},
    )
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
        case _:
            raise ValueError(f"unknown agent: {args.agent}")


if __name__ == "__main__":
    asyncio.run(main())
