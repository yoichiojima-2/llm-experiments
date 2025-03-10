import os
from dataclasses import dataclass
from logging import getLogger
from typing import Literal

import langsmith
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables.base import Runnable
from langgraph.graph import MessagesState
from langgraph.types import Command
from pydantic import BaseModel, Field

import agents
from utils import parse_base_model, get_last_message

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
    "__end__",
]

SUPERVISOR_LITERAL = Literal[
    "spotify",
    "shell",
    "python",
    "duckduckgo",
    "wikipedia",
    "browser",
    "files",
    "__end__",
]


@dataclass
class Node:
    model: BaseChatModel
    agent: Runnable

    def node(self):
        async def f(state: MessagesState) -> Command[Literal["supervisor"]]:
            res = await self.agent.ainvoke({"messages": [get_last_message(state)]})
            return Command(goto="supervisor", update={"messages": [get_last_message(res).content]})

        return f

    @classmethod
    def new(cls, model, agent, *a, **kw):
        return cls(model, agent().agent(model, *a, **kw))


class SupervisorNode(Node):
    # todo: remove __end__, spotify duplicates
    def __init__(self, model, *a, **kw):
        self.model = model
        self.agent = agents.SupervisorAgent().agent(self.model, *a, **kw)

    class Output(BaseModel):
        next_agent: SUPERVISOR_LITERAL = Field(description="The next agent to invoke")

    def node(self):
        async def f(
            state: MessagesState,
        ) -> Command[SUPERVISOR_LITERAL]:
            c = langsmith.Client(api_key=os.getenv("LANGSMITH_API_KEY"))
            prompt = c.pull_prompt("homanp/superagent")
            chain = prompt | self.model.with_structured_output(self.Output)
            payload = {
                "input": get_last_message(state),
                "output_format": parse_base_model(self.Output),
                "tools": INSTALLED_AGENTS,
            }
            res = await chain.ainvoke(payload)
            print(f"supervisor: {res.next_agent}")
            return Command(goto=res.next_agent)

        return f


class SpotifyNode(Node):
    def __init__(self, model, *a, **kw):
        self.model = model
        self.agent = agents.SpotifyAgent().agent(self.model, *a, **kw)

    def node(self):
        async def f(state: MessagesState) -> Command[Literal["supervisor"]]:
            payload = {"input": [get_last_message(state)]}
            res = await self.agent.ainvoke(payload)
            return Command(goto="supervisor", update={"messages": [res["output"]]})

        return f
