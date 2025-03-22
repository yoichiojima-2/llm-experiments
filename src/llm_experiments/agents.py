import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import yaml
from langchain.agents import AgentType, initialize_agent
from langchain_community.agent_toolkits import FileManagementToolkit, PlayWrightBrowserToolkit
from langchain_community.agent_toolkits.openapi import planner
from langchain_community.agent_toolkits.openapi.spec import reduce_openapi_spec
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.tools import DuckDuckGoSearchRun, ShellTool, WikipediaQueryRun
from langchain_community.utilities import GoogleSerperAPIWrapper, WikipediaAPIWrapper
from langchain_community.utilities.requests import RequestsWrapper
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import Tool
from langchain_experimental.utilities import PythonREPL
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent
from playwright.async_api import Playwright
from spotipy import util as spotipy_util
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool


@dataclass
class Agent(ABC):
    model: BaseChatModel

    @abstractmethod
    def agent(self): ...

    @staticmethod
    def get_last_response(res):
        return res["messages"][-1].content


class OpenAPIAgent(Agent):
    def __init__(self, model, spec_filename):
        self.model = model
        spec_dir = Path(__file__).parent.parent.parent / "openapi"
        with Path(spec_dir / spec_filename).open() as f:
            self.spec_raw = yaml.load(f, Loader=yaml.Loader)
            self.spec_reduced = reduce_openapi_spec(self.spec_raw)

    @property
    def auth_header(self):
        scopes = list(
            self.spec_raw["components"]["securitySchemes"]["oauth_2_0"]["flows"]["authorizationCode"]["scopes"].keys()
        )
        access_token = spotipy_util.prompt_for_user_token(scope=",".join(scopes))
        return {"Authorization": f"Bearer {access_token}"}

    def agent(self, handle_parsing_errors=True, *a, **kw):
        return planner.create_openapi_agent(
            self.spec_reduced,
            RequestsWrapper(headers=self.auth_header),
            self.model,
            handle_parsing_errors=handle_parsing_errors,
            allow_dangerous_requests=True,  # fixme
            *a,
            **kw,
        )

    @staticmethod
    def get_last_response(res):
        return res["output"]


class SpotifyAgent(Agent):
    def agent(self, handle_parsing_errors=True, *a, **kw):
        return OpenAPIAgent(self.model, "spotify.yml").agent(handle_parsing_errors=handle_parsing_errors, *a, **kw)

    @staticmethod
    def get_last_response(res):
        return OpenAPIAgent.get_last_response(res)


class DuckDuckGoAgent(Agent):
    def agent(self, *a, **kw):
        self.tools = [DuckDuckGoSearchRun()]
        return create_react_agent(self.model, tools=self.tools, *a, **kw)


class ShellAgent(Agent):
    def agent(self, *a, **kw):
        self.tools = [ShellTool()]
        return create_react_agent(self.model, tools=self.tools, *a, **kw)


@dataclass
class BrowserAgent(Agent):
    model: BaseChatModel
    playwright: Playwright

    async def agent(self, *a, **kw):
        browser = await self.playwright.chromium.launch(headless=False)
        tools = PlayWrightBrowserToolkit.from_browser(async_browser=browser).get_tools()
        return initialize_agent(
            tools,
            self.model,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            *a,
            **kw,
        )

    @staticmethod
    def get_last_response(res):
        return res["output"]


class PythonAgent(Agent):
    def agent(self, *a, **kw):
        self.tools = [
            Tool(
                name="python_repl",
                func=PythonREPL().run,
                description="Python REPL",
            )
        ]
        return create_react_agent(self.model, tools=self.tools, *a, **kw)


class WikipediaAgent(Agent):
    def agent(self, *a, **kw):
        self.tools = [WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())]
        return create_react_agent(self.model, tools=self.tools, *a, **kw)


class FileAgent(Agent):
    def agent(self, *a, **kw):
        self.tools = FileManagementToolkit().get_tools()
        return create_react_agent(self.model, tools=self.tools, *a, **kw)


class SerperAgent(Agent):
    def agent(self, *a, **kw):
        self.tools = [GoogleSerperAPIWrapper().run]
        return create_react_agent(self.model, tools=self.tools, *a, **kw)


class TavilyAgent(Agent):
    def agent(self, *a, **kw):
        self.tools = [TavilySearch(max_results=5)]
        return create_react_agent(self.model, tools=self.tools, *a, **kw)


@dataclass
class SQLAgent(Agent):
    db_name: str

    def agent(self, *a, **kw):
        db_dir = Path(__file__).parent.parent.parent / "db"
        path = db_dir / f"{self.db_name}.db"
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path, check_same_thread=False)
        engine = create_engine(
            f"sqlite://{path}",
            creator=lambda: conn,
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )
        db = SQLDatabase(engine)
        self.tools = SQLDatabaseToolkit(db=db, llm=self.model).get_tools()
        return create_react_agent(self.model, tools=self.tools, *a, **kw)
