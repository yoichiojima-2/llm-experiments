import asyncio
import os
from dataclasses import dataclass
from logging import getLogger
from typing import Literal

import langsmith
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables.base import Runnable
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from pydantic import BaseModel, Field

import agents
from utils import get_last_message, parse_base_model

logger = getLogger(__name__)
logger.setLevel("DEBUG")

INSTALLED_AGENTS = [
    "spotify",
    "shell",
    "python",
    "duckduckgo",
    "wikipedia",
    "browser",
    "files",
    "serper",
    "tavily",
    "sql",
    "user",
]

SUPERVISOR_LITERAL = Literal[
    "spotify",
    "shell",
    "python",
    "duckduckgo",
    "wikipedia",
    "browser",
    "files",
    "serper",
    "tavily",
    "sql",
    "user",
]


@dataclass
class Node:
    model: BaseChatModel
    agent: Runnable

    def node(self):
        async def f(state: BaseModel) -> Command[Literal["supervisor"]]:
            res = await self.agent.ainvoke({"messages": [get_last_message(state)]})
            return Command(goto="supervisor", update={"messages": [get_last_message(res).content]})

        return f

    @classmethod
    def new(cls, model, agent, *a, **kw):
        return cls(model, agent().agent(model, *a, **kw))


class SupervisorNode(Node):
    def __init__(self, model, *a, **kw):
        self.model = model
        self.agent = create_react_agent(model, [], *a, **kw)

    class Output(BaseModel):
        next_agent: SUPERVISOR_LITERAL = Field(description="The next agent to invoke")
        message: str = Field(description="The message to display")

    def node(self):
        async def f(state: BaseModel) -> Command[SUPERVISOR_LITERAL]:
            c = langsmith.Client(api_key=os.getenv("LANGSMITH_API_KEY"))
            prompt = c.pull_prompt("homanp/superagent")
            chain = prompt | self.model.with_structured_output(self.Output)
            payload = {
                "input": get_last_message(state),
                "output_format": parse_base_model(self.Output),
                "tools": INSTALLED_AGENTS,
            }
            res = await chain.ainvoke(payload)
            print(f"supervisor: [{res.next_agent}] {res.message}")
            return Command(goto=res.next_agent, update={"messages": [res.message]})

        return f


class UserNode(Node):
    def __init__(self, model, *a, **kw):
        self.model = model

    @staticmethod
    async def interrupt(prompt: str) -> str:
        return await asyncio.to_thread(input, prompt)

    def node(self):
        async def f(_: BaseModel) -> Command[Literal["supervisor", "__end__"]]:
            user_input = await self.interrupt("input: ")
            if user_input == "exit":
                return Command(goto="__end__", update={"messages": ["bye"]})
            else:
                return Command(goto="supervisor", update={"messages": [user_input]})

        return f


class SpotifyNode(Node):
    def __init__(self, model, *a, **kw):
        self.model = model
        self.agent = agents.SpotifyAgent().agent(self.model, *a, **kw)

    def node(self):
        async def f(state: BaseModel) -> Command[Literal["supervisor"]]:
            payload = {"input": [get_last_message(state)]}
            res = await self.agent.ainvoke(payload)
            return Command(goto="supervisor", update={"messages": [res["output"]]})

        return f
