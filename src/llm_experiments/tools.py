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


def duckduckgo():
    return DuckDuckGoSearchRun()


def shell(*a, **kw):
    return ShellTool(*a, **kw)


async def browser_tools(browser):
    import nest_asyncio

    nest_asyncio.apply()
    return PlayWrightBrowserToolkit.from_browser(async_browser=browser).get_tools()


def python_repl() -> str:
    @tool
    def python_repl_tool(script):
        """
        python repr run
        """
        return PythonREPL().run(script)

    return python_repl_tool


def wikipedia():
    return WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())


def file_management_tools():
    return FileManagementToolkit().get_tools()


def serper():
    @tool
    def serper_tool(query) -> str:
        """
        serper search
        """
        return GoogleSerperAPIWrapper().run(query)

    return serper_tool


def tavily():
    return TavilySearch(max_results=5)


def sql_tools(model, db_name):
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
