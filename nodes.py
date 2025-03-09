import os
from abc import ABC, abstractmethod
import typing as t

import langsmith
from langgraph.types import Command
from pydantic import BaseModel, Field
from langgraph.types import Command
from langgraph.graph import StateGraph, MessagesState, START, END

import agents
from langgraph.types import Command
from utils import parse_base_model


class Node(ABC):
    @abstractmethod
    def node(self): ...

    def get_last_message(self, state):
        return state["messages"][-1]


class SupervisorNode(Node):
    def __init__(self, model):
        self.model = model

    class Output(BaseModel):
        next_agent: str = Field(description="The next agent to invoke")

    def node(self):
        async def f(state: MessagesState) -> Command[t.Literal[END, "spotify"]]:
            c = langsmith.Client(api_key=os.getenv("LANGSMITH_API_KEY"))
            prompt = c.pull_prompt("homanp/superagent")
            chain = prompt | self.model.with_structured_output(self.Output)
            res = await chain.ainvoke(
                {
                    "input": self.get_last_message(state),
                    "output_format": parse_base_model(self.Output),
                    "tools": agents.__all__,
                }
            )
            return Command(goto=res.next_agent)
        return f


class SpotifyNode(Node):
    def __init__(self, model):
        self.model = model
        self.agent = agents.SpotifyAgent().agent(self.model)

    def node(self):
        async def f(state: MessagesState) -> Command[t.Literal["supervisor"]]:
            res = await self.agent().ainvoke(self.get_last_message(state))
            return Command(goto="supervisor", update={"messages": [res]})
        return f
