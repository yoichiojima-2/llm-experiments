import operator
import os
from abc import ABC, abstractmethod
from logging import getLogger
from typing import Annotated, Literal

import langsmith
from langgraph.graph import MessagesState
from langgraph.types import Command
from pydantic import BaseModel, Field

import agents
from utils import parse_base_model

logger = getLogger(__name__)
logger.setLevel("DEBUG")


class State(MessagesState):
    scratchpad: Annotated[str, operator.add] = Field(
        description="A scratchpad for the user to write notes"
    )


class Node(ABC):
    @abstractmethod
    def node(self): ...

    def get_last_message(self, state):
        return state["messages"][-1]


class SupervisorNode(Node):
    # todo: remove __end__, spotify duplicates
    def __init__(self, model, *a, **kw):
        self.model = model
        self.agent = agents.SupervisorAgent().agent(self.model, *a, **kw)

    class Output(BaseModel):
        next_agent: Literal["__end__", "spotify", "shell"] = Field(
            description="The next agent to invoke"
        )

    def node(self):
        async def f(
            state: MessagesState,
        ) -> Command[Literal["__end__", "spotify", "shell"]]:
            c = langsmith.Client(api_key=os.getenv("LANGSMITH_API_KEY"))
            prompt = c.pull_prompt("homanp/superagent")
            chain = prompt | self.model.with_structured_output(self.Output)
            payload = {
                "input": self.get_last_message(state),
                "output_format": parse_base_model(self.Output),
                "tools": ["__end__", "spotify", "shell"],
            }
            res = await chain.ainvoke(payload)
            return Command(goto=res.next_agent)

        return f


class SpotifyNode(Node):
    def __init__(self, model, *a, **kw):
        self.model = model
        self.agent = agents.SpotifyAgent().agent(self.model, *a, **kw)

    def node(self):
        async def f(state: State) -> Command[Literal["supervisor"]]:
            payload = {"input": [self.get_last_message(state)]}
            res = await self.agent.ainvoke(payload)
            return Command(goto="supervisor", update={"messages": [res["output"]]})

        return f


class ShellNode(Node):
    def __init__(self, model, *a, **kw):
        self.model = model
        self.agent = agents.ShellAgent().agent(self.model, *a, **kw)

    def node(self):
        async def f(state: State) -> Command[Literal["supervisor"]]:
            payload = {"input": [self.get_last_message(state)]}
            res = await self.agent.ainvoke(payload)
            msg = res["messages"][-1]
            return Command(goto="supervisor", update={"messages": [msg.content]})

        return f
