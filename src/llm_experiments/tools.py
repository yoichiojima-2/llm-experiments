import sqlite3
from pathlib import Path

from langchain_community.agent_toolkits import FileManagementToolkit, PlayWrightBrowserToolkit
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.tools import DuckDuckGoSearchRun, ShellTool, WikipediaQueryRun
from langchain_community.utilities import GoogleSerperAPIWrapper, WikipediaAPIWrapper
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.tools import tool
from langchain_core.tools.base import BaseTool
from langchain_experimental.utilities import PythonREPL
from langchain_tavily import TavilySearch
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from llm_experiments.custom_tools.slack import SlackToolkit


def duckduckgo() -> list[BaseTool]:
    return [DuckDuckGoSearchRun()]


def shell(*a, **kw) -> list[BaseTool]:
    return [ShellTool(*a, **kw)]


def python_repl() -> list[BaseTool]:
    @tool
    def python_repl_tool(script):
        """python repr run"""
        return PythonREPL().run(script)

    return [python_repl_tool]


def wikipedia() -> list[BaseTool]:
    return [WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())]


def serper() -> list[BaseTool]:
    @tool
    def serper_tool(query):
        """serper search"""
        return GoogleSerperAPIWrapper().run(query)

    return [serper_tool]


def tavily() -> list[BaseTool]:
    return [TavilySearch(max_results=5)]


async def browser(browser) -> list[BaseTool]:
    return PlayWrightBrowserToolkit.from_browser(async_browser=browser).get_tools()


def slack() -> list[BaseTool]:
    return SlackToolkit().get_tools()


def file_management(*a, **kw) -> list[BaseTool]:
    return FileManagementToolkit(*a, **kw).get_tools()


def sql(model, db_name) -> list[BaseTool]:
    db_dir = Path(__file__).parent.parent.parent / "db"
    path = db_dir / f"{db_name}.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    engine = create_engine(
        f"sqlite://{path}",
        creator=lambda: conn,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    db = SQLDatabase(engine)
    return SQLDatabaseToolkit(db=db, llm=model).get_tools()


# utils
def make_tools_by_name(tools) -> dict[str, BaseTool]:
    return {tool.name: tool for tool in tools}
