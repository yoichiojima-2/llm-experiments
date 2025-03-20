import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path

import yaml
from langchain.agents import AgentType, initialize_agent
from langchain_community.agent_toolkits import FileManagementToolkit, PlayWrightBrowserToolkit
from langchain_community.agent_toolkits.openapi import planner
from langchain_community.agent_toolkits.openapi.spec import reduce_openapi_spec
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.tools import DuckDuckGoSearchRun, ShellTool, TavilySearchResults, WikipediaQueryRun
from langchain_community.tools.playwright.utils import create_async_playwright_browser
from langchain_community.utilities import GoogleSerperAPIWrapper, WikipediaAPIWrapper
from langchain_community.utilities.requests import RequestsWrapper
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.tools import Tool
from langchain_experimental.utilities import PythonREPL
from langgraph.prebuilt import create_react_agent
from spotipy import util as spotipy_util
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

__all__ = [
    "SpotifyAgent",
    "DuckDuckGoAgent",
    "ShellAgent",
    "BrowserAgent",
    "PythonAgent",
    "WikipediaAgent",
    "FileAgent",
]

DB_DIR = Path(__file__).parent / "db"


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
            self.spec_raw["components"]["securitySchemes"]["oauth_2_0"]["flows"]["authorizationCode"]["scopes"].keys()
        )
        access_token = spotipy_util.prompt_for_user_token(scope=",".join(scopes))
        return {"Authorization": f"Bearer {access_token}"}

    def agent(self, model, handle_parsing_errors=True, *a, **kw):
        return planner.create_openapi_agent(
            self.spec_reduced,
            RequestsWrapper(headers=self.auth_header),
            model,
            handle_parsing_errors=handle_parsing_errors,
            allow_dangerous_requests=True,  # fixme
            *a,
            **kw,
        )


class SpotifyAgent(Agent):
    def agent(self, model, handle_parsing_errors=True, *a, **kw):
        return OpenAPIAgent(Path(__file__).parent.parent.parent / "openapi/spotify.yml").agent(
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
    async def agent(self, model, *a, **kw):
        browser = await create_async_playwright_browser()
        tools = PlayWrightBrowserToolkit.from_browser(async_browser=browser).get_tools()
        return initialize_agent(
            tools,
            model,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            *a,
            **kw,
        )


class PythonAgent(Agent):
    def agent(self, model, *a, **kw):
        self.tools = [
            Tool(
                name="python_repl",
                func=PythonREPL().run,
                description="Python REPL",
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


class SerperAgent(Agent):
    def agent(self, model, *a, **kw):
        self.tools = [GoogleSerperAPIWrapper().run]
        return create_react_agent(model, tools=self.tools, *a, **kw)


class TavilyAgent(Agent):
    def agent(self, model, *a, **kw):
        self.tool = TavilySearchResults(
            max_results=5,
            search_depth="advanced",
            include_answer=True,
            include_raw_content=True,
            include_images=True,
        )
        return create_react_agent(model, tools=self.tools, *a, **kw)


class SQLAgent(Agent):
    def agent(self, model, db_name, *a, **kw):
        path = DB_DIR / f"{db_name}.db"
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path, check_same_thread=False)
        engine = create_engine(
            f"sqlite://{path}",
            creator=lambda: conn,
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )
        db = SQLDatabase(engine)
        self.tools = SQLDatabaseToolkit(db=db, llm=model).get_tools()
        return create_react_agent(model, tools=self.tools, *a, **kw)
