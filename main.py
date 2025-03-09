import asyncio
from abc import ABC, abstractmethod
from argparse import ArgumentParser
from pathlib import Path

import yaml
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_community.agent_toolkits import (
    FileManagementToolkit,
    PlayWrightBrowserToolkit,
)
from langchain_community.agent_toolkits.openapi import planner
from langchain_community.agent_toolkits.openapi.spec import reduce_openapi_spec
from langchain_community.tools import DuckDuckGoSearchRun, ShellTool, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.utilities.requests import RequestsWrapper
from langchain_core.tools import Tool
from langchain_experimental.utilities import PythonREPL
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from playwright.async_api import async_playwright
from spotipy import util as spotipy_util

load_dotenv()


def parse_args():
    parser = ArgumentParser()
    opt = parser.add_argument
    opt("-q", "--query", required=True)
    return parser.parse_args()


class Playwright:
    async def __aenter__(self):
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=False)
        toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=self.browser)
        self.tools = toolkit.get_tools()
        return self

    async def __aexit__(self, *a):
        await self.browser.close()
        await self.playwright.stop()


class Agent(ABC):
    @abstractmethod
    def agent(self): ...


class OpenAPIAgent(Agent):
    def __init__(self, spec_path):
        with Path(spec_path).open() as f:
            self.spec_raw = yaml.load(f, Loader=yaml.Loader)
            self.spec_reduced = reduce_openapi_spec(self.spec_raw)

    @property
    def auth_header(self):
        scopes = list(
            self.spec_raw["components"]["securitySchemes"]["oauth_2_0"]["flows"][
                "authorizationCode"
            ]["scopes"].keys()
        )
        access_token = spotipy_util.prompt_for_user_token(scope=",".join(scopes))
        return {"Authorization": f"Bearer {access_token}"}

    def agent(self, model, *a, **kw):
        return planner.create_openapi_agent(
            self.spec_reduced,
            RequestsWrapper(headers=self.auth_header),
            model,
            allow_dangerous_requests=True,  # fixme
            *a,
            **kw,
        )


class SpotifyAgent(Agent):
    def agent(self, model, handle_parsing_errors=True, *a, **kw):
        return OpenAPIAgent("openapi/spotify.yml").agent(
            model, handle_parsing_errors=handle_parsing_errors, *a, **kw
        )


class DuckDuckGoAgent(Agent):
    def agent(self, model, *a, **kw):
        self.tools = [DuckDuckGoSearchRun()]
        return create_react_agent(model, tools=self.tools, *a, **kw)


class ShellAgent(Agent):
    def agent(self, model, *a, **kw):
        self.tools = [ShellTool()]
        return create_react_agent(model, tools=self.tools, *a, **kw)


class BrowserAgent(Agent):
    def agent(self, model, playwright, *a, **kw):
        self.tools = playwright.tools
        return create_react_agent(model, tools=self.tools, *a, **kw)


class PythonAgent(Agent):
    def agent(self, model, *a, **kw):
        self.tools = [
            Tool(
                name="python_repl",
                description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
                func=PythonREPL().run,
            )
        ]
        return create_react_agent(model, tools=self.tools, *a, **kw)


class WikipediaAgent(Agent):
    def agent(self, model, *a, **kw):
        self.tools = [WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())]
        return create_react_agent(model, tools=self.tools, *a, **kw)


class FileAgent(Agent):
    def agent(self, model, *a, **kw):
        self.tools = FileManagementToolkit().get_tools()
        return create_react_agent(model, tools=self.tools, *a, **kw)


async def print_stream(stream):
    async for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()


async def run(query, thread_id="1"):
    inputs = {"messages": [("user", query)]}
    config = {"configurable": {"thread_id": thread_id}}

    model = init_chat_model("gpt-4o-mini", model_provider="openai")

    async with Playwright() as _:
        memory = MemorySaver()
        agent = FileAgent().agent(model, checkpointer=memory)

        res = agent.astream(inputs, config, stream_mode="values")
        await print_stream(res)


async def main():
    args = parse_args()
    await run(args.query)


if __name__ == "__main__":
    asyncio.run(main())
