from abc import ABC, abstractmethod
from pathlib import Path

import yaml
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_community.agent_toolkits.openapi import planner
from langchain_community.agent_toolkits.openapi.spec import reduce_openapi_spec
from langchain_community.tools import DuckDuckGoSearchRun, ShellTool, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.utilities.requests import RequestsWrapper
from langchain_core.tools import Tool
from langchain_experimental.utilities import PythonREPL
from langgraph.prebuilt import create_react_agent
from spotipy import util as spotipy_util

__all__ = [
    "SpotifyAgent",
    "DuckDuckGoAgent",
    "ShellAgent",
    "BrowserAgent",
    "PythonAgent",
    "WikipediaAgent",
    "FileAgent",
]


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
