import os
from abc import ABC

import langsmith
from langgraph.types import Command
from pydantic import BaseModel, Field

from utils import parse_base_model


class Node(ABC):
    Output = BaseModel

    def node(self): ...


class Supervisor(Node):
    def __init__(self, model, agents):
        self.model = model
        self.agents = agents

    class Output(BaseModel):
        next_agent: str = Field(description="The next agent to invoke")

    async def node(self, state):
        c = langsmith.Client(api_key=os.getenv("LANGSMITH_API_KEY"))
        prompt = c.pull_prompt("homanp/superagent")
        chain = prompt | self.model.with_structured_output(self.Output)
        res = await chain.ainvoke(
            {
                "input": state["query"],
                "output_format": parse_base_model(self.Output),
                "tools": self.agents,
            }
        )
        return Command(goto=res.next_agent)
