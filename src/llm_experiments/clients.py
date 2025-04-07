import asyncio
import sys
from argparse import ArgumentParser, Namespace

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools.playwright.utils import create_async_playwright_browser
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.memory import BaseCheckpointSaver, MemorySaver
from langgraph.graph.graph import CompiledGraph

from llm_experiments import prompts, tools
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
    opt("--verbose", "-v", action="store_true", default=False)
    return parser.parse_args()


async def main() -> None:
    load_dotenv()
    args = parse_args()
    config = Config(
        model=create_model(args.model),
        memory=MemorySaver(),
        verbose=args.verbose,
        configurable={"configurable": {"thread_id": "client"}},
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
        case "slack":
            await slack(config)
        case "python-repl":
            await python_repl(config)
        case _:
            raise ValueError(f"unknown agent: {args.agent}")


class Config:
    def __init__(
        self, model: BaseChatModel, memory: BaseCheckpointSaver, verbose: bool, configurable: dict[str, dict[str, str]]
    ):
        self.model = model
        self.memory = memory
        self.configurable = configurable
        self.verbose = verbose


def create_executor(model, tools, verbose, prompt) -> AgentExecutor:
    agent = create_react_agent(llm=model, tools=tools, prompt=prompt)
    return AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True, verbose=verbose)


async def interactive_chat(agent: CompiledGraph) -> None:
    try:
        while True:
            user_input = input("user: ")
            if user_input == "q":
                print("quitting...")
                return

            print()
            async for i in agent.graph.astream({"messages": [user_input]}, config=agent.config, stream_mode="messages"):
                print(i[0].content, end="")
            print("\n\n")
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)


async def interactive_chat_from_tools(tools, config):
    prompt = prompts.multipurpose()
    executor = create_executor(model=config.model, tools=tools, verbose=config.verbose, prompt=prompt)
    agent = Agent(
        executor=executor,
        model=config.model,
        memory=config.memory,
        tools=tools,
        verbose=config.verbose,
        config=config.configurable,
    )
    await interactive_chat(agent)


async def search(config: Config) -> None:
    toolkit = [tools.tavily(), tools.duckduckgo(), tools.serper(), tools.wikipedia()]
    await interactive_chat_from_tools(tools=toolkit, config=config)


async def shell_w_search(config: Config) -> None:
    toolkit = [tools.shell(ask_human_input=True), tools.tavily(), tools.duckduckgo(), tools.serper()]
    await interactive_chat_from_tools(tools=toolkit, config=config)


async def shell(config: Config) -> None:
    toolkit = [tools.shell(ask_human_input=True)]
    await interactive_chat_from_tools(tools=toolkit, config=config)


# fixme
async def slack(config: Config) -> None:
    toolkit = [*tools.slack_tools()]
    await interactive_chat_from_tools(tools=toolkit, config=config)


async def python_repl(config: Config) -> None:
    toolkit = [tools.python_repl()]
    await interactive_chat_from_tools(tools=toolkit, config=config)


async def sql(config: Config) -> None:
    toolkit = [*tools.sql_tools(config.model, "sql"), tools.shell(), tools.duckduckgo()]
    await interactive_chat_from_tools(tools=toolkit, config=config)


# fixme
async def browser(config: Config) -> None:
    import nest_asyncio

    nest_asyncio.apply()

    async with create_async_playwright_browser(headless=False) as async_browser:
        toolkit = await tools.browser_tools(async_browser)
        await interactive_chat_from_tools(tools=toolkit, config=config)


if __name__ == "__main__":
    asyncio.run(main())
