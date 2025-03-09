import os
from abc import ABC, abstractmethod
from typing import Literal

import langsmith
from langgraph.graph import END, MessagesState
from langgraph.types import Command
from pydantic import BaseModel, Field

import agents
from utils import parse_base_model


class Node(ABC):
    @abstractmethod
    def node(self): ...

    def get_last_message(self, state):
        return state["messages"][-1]


class SupervisorNode(Node):
    # todo: remove __end__, spotify duplicates
    def __init__(self, model):
        self.model = model

    class Output(BaseModel):
        next_agent: Literal["__end__", "spotify"] = Field(description="The next agent to invoke")

    def node(self):
        async def f(state: MessagesState) -> Command[Literal["__end__", "spotify"]]:
            c = langsmith.Client(api_key=os.getenv("LANGSMITH_API_KEY"))
            prompt = c.pull_prompt("homanp/superagent")
            chain = prompt | self.model.with_structured_output(self.Output)
            payload = {
                "input": self.get_last_message(state),
                "output_format": parse_base_model(self.Output),
                "tools": ["__end__", "spotify"],
            }
            print(f"supervisor: {prompt.invoke(payload)}")
            res = await chain.ainvoke(payload)
            print(f"supervisor: {res.next_agent}")
            return Command(goto=res.next_agent)

        return f


class SpotifyNode(Node):
    def __init__(self, model):
        self.model = model
        self.agent = agents.SpotifyAgent().agent(self.model)

    def node(self):
        async def f(state: MessagesState) -> Command[t.Literal["supervisor"]]:
            payload = {"input": [self.get_last_message(state)]}
            res = await self.agent.ainvoke(payload)
            return Command(goto="supervisor", update={"messages": [res["output"]]})

        return f
