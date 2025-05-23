import os
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import slack_sdk
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.tools import DuckDuckGoSearchRun, ShellTool, WikipediaQueryRun
from langchain_community.utilities import GoogleSerperAPIWrapper, WikipediaAPIWrapper
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import tool
from langchain_core.tools.base import BaseTool
from langchain_core.tools.structured import StructuredTool
from langchain_experimental.utilities import PythonREPL
from langchain_tavily import TavilySearch
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool


class Tools(ABC):
    @abstractmethod
    def get_tools(self) -> list[BaseTool]: ...

    def get_tools_by_name(self) -> dict[str, BaseTool]:
        return tools_by_name(self.get_tools())


class DuckDuckGo(Tools):
    def get_tools(self, *a, **kw) -> list[BaseTool]:
        @tool
        def search_duckduckgo(query):
            """DuckDuckGo search"""
            return DuckDuckGoSearchRun(*a, **kw).run(query)

        return [search_duckduckgo]


class Shell(Tools):
    def get_tools(self, *a, **kw) -> list[BaseTool]:
        @tool
        def run_shell(command: str):
            """run shell command"""
            return ShellTool(*a, **kw).run(command)

        return [run_shell]


class Python_(Tools):
    def get_tools(self, *a, **kw) -> list[BaseTool]:
        @tool
        def run_python(script):
            """run python"""
            return PythonREPL(*a, **kw).run(script)

        return [run_python]


class Wikipedia(Tools):
    def get_tools(self, *a, **kw) -> list[BaseTool]:
        @tool
        def search_wikipedia(query):
            """search for wikipedia"""
            return WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(), *a, **kw).run(query)

        return [search_wikipedia]


class Serper(Tools):
    def get_tools(self, *a, **kw) -> list[BaseTool]:
        @tool
        def search_serper(query):
            """search for serper"""
            return GoogleSerperAPIWrapper(*a, **kw).run(query)

        return [search_serper]


@dataclass
class Tavily(Tools):
    max_results: int = 5

    def get_tools(self, *a, **kw) -> list[BaseTool]:
        @tool
        def search_tavily(query):
            """search for tavily"""
            return TavilySearch(max_results=self.max_results, *a, **kw).run(query)

        return [search_tavily]


class Browser(Tools):
    def __init__(self, browser):
        self.browser = browser

    def get_tools(self, *a, **kw) -> list[BaseTool]:
        return [
            base_to_structured(i)
            for i in PlayWrightBrowserToolkit.from_browser(async_browser=self.browser, *a, **kw).get_tools()
        ]


class SQL(Tools):
    def __init__(self, llm: BaseChatModel, db_name: str = "db"):
        self.llm = llm
        self.db = self._create_db(db_name)

    def _create_db(self, db_name: str) -> SQLDatabase:
        db_dir = Path(__file__).parent.parent.parent / "db"
        path = db_dir / f"{db_name}.db"
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path, check_same_thread=False)
        engine = create_engine(
            f"sqlite:///{path}",
            creator=lambda: conn,
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )
        return SQLDatabase(engine)

    def get_tools(self, *a, **kw) -> list[BaseTool]:
        return [base_to_structured(i) for i in SQLDatabaseToolkit(llm=self.llm, db=self.db, *a, **kw).get_tools()]


class Slack(Tools):
    @property
    def client(self):
        return slack_sdk.WebClient(token=os.getenv("SLACK_USER_TOKEN"))

    def get_tools(self) -> list[BaseTool]:
        @tool
        def post_message(channel: str, text: str):
            """post a message to a slack channel."""
            return self.client.chat_postMessage(channel=channel, text=text)

        @tool
        def delete_message(channel: str, ts: str):
            """delete a message from a slack channel."""
            return self.client.chat_delete(channel=channel, ts=ts)

        @tool
        def post_ephemeral(channel: str, text: str, user: str):
            """post an ephemeral message to a slack channel."""
            return self.client.chat_postEphemeral(channel=channel, text=text, user=user)

        @tool
        def update_message(channel: str, ts: str, text: str):
            """update a message in a slack channel."""
            return self.client.chat_update(channel=channel, ts=ts, text=text)

        @tool
        def add_reaction(channel: str, emoji_name: str, ts: str):
            """add a reaction to a message in a slack channel."""
            return self.client.reactions_add(channel=channel, name=emoji_name, timestamp=ts)

        @tool
        def remove_reaction(channel: str, emoji_name: str, ts: str):
            """remove a reaction from a message in a slack channel."""
            return self.client.reactions_remove(channel=channel, name=emoji_name, timestamp=ts)

        @tool
        def upload_file(channels: str, file: str):
            """upload a file to a slack channel."""
            return self.client.files_upload_v2(channels=channels, file=file)

        @tool
        def add_remote_file(channels: list[str], file: str):
            """add a remote file to a slack channel."""
            return self.client.files_remote_add(channels=channels, file=file)

        @tool
        def list_conversations(limit: int = 100):
            """list all conversations in slack."""
            return self.client.conversations_list(limit=limit)

        @tool
        def get_conversation_history(channel: str):
            """get the history of a conversation in slack."""
            return self.client.conversations_history(channel=channel)

        @tool
        def start_direct_message(users: list[str]):
            """open a conversation in slack."""
            return self.client.conversations_open(users=users)

        @tool
        def create_channel(name: str):
            """create a new channel in slack."""
            return self.client.conversations_archive(name=name)

        @tool
        def get_conversation_info(channel: str):
            """get information about a conversation in slack."""
            return self.client.conversations_info(channel=channel)

        @tool
        def get_members_of_conversation(channel: str):
            """get members of a conversation in slack."""
            return self.client.conversations_members(channel=channel)

        @tool
        def join_conversation(channel: str):
            """join a conversation in slack."""
            return self.client.conversations_join(channel=channel)

        @tool
        def leave_conversation(channel: str):
            """leave a conversation in slack."""
            return self.client.conversations_leave(channel=channel)

        return [
            post_message,
            delete_message,
            post_ephemeral,
            update_message,
            add_reaction,
            remove_reaction,
            upload_file,
            add_remote_file,
            list_conversations,
            get_conversation_history,
            start_direct_message,
            create_channel,
            get_conversation_info,
            get_members_of_conversation,
            join_conversation,
            leave_conversation,
        ]


def base_to_structured(tool: BaseTool) -> StructuredTool:
    return StructuredTool(description=tool.description, name=tool.name, func=tool._run, args_schema=tool.args_schema)


def tools_by_name(tools: list[BaseTool]) -> dict[str, BaseTool]:
    return {tool.name: tool for tool in tools}
