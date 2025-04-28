import sqlite3
from pathlib import Path

from langchain_community.agent_toolkits import FileManagementToolkit, PlayWrightBrowserToolkit
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.tools import DuckDuckGoSearchRun, ShellTool, WikipediaQueryRun
from langchain_community.utilities import GoogleSerperAPIWrapper, WikipediaAPIWrapper
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from langchain_tavily import TavilySearch
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from langchain_core.tools.base import BaseTool

from llm_experiments.custom_tools.slack import SlackToolkit



# singuler tools
def duckduckgo() -> BaseTool:
    return DuckDuckGoSearchRun()


def shell(*a, **kw) -> BaseTool:
    return ShellTool(*a, **kw)


def python_repl() -> BaseTool:
    @tool
    def python_repl_tool(script):
        """python repr run"""
        return PythonREPL().run(script)

    return python_repl_tool


def wikipedia() -> BaseTool:
    return WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())


def serper() -> BaseTool:
    @tool
    def serper_tool(query):
        """serper search"""
        return GoogleSerperAPIWrapper().run(query)

    return serper_tool


def tavily() -> BaseTool:
    return TavilySearch(max_results=5)


# toolkits
async def browser_tools(browser) -> list[BaseTool]:
    return PlayWrightBrowserToolkit.from_browser(async_browser=browser).get_tools()


def slack_tools() -> list[BaseTool]:
    return SlackToolkit().get_tools()


def file_management_tools(*a, **kw) -> list[BaseTool]:
    return FileManagementToolkit(*a, **kw).get_tools()


def sql_tools(model, db_name) -> BaseTool:
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
