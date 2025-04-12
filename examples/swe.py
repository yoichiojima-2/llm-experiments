"""
this example demonstrates how to create a software development team
"""

import asyncio
import sys
from typing import Literal

from langchain_core.tools import tool as member  # just don't like call members tools
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode as MemberNode  # just don't like call members tools
from langgraph.types import Command

from llm_experiments import tools as t
from llm_experiments.agent import Agent
from llm_experiments.llm import create_model


async def main():
    config = {"configurable": {"thread_id": "software-development"}}
    dev_team = SWE_Team(create_model(), MemorySaver(), config)
    await dev_team.start_interactive_chat()


class SWE_Team:
    def __init__(self, model, memory, config):
        self.model = model
        self.memory = memory
        self.config = config
        self.members = [self.designer_node, self.programmer_node, self.reviewer_node, self.tester_node]
        self.graph = self.compile_graph()

    def compile_graph(self):
        g = StateGraph(MessagesState)
        g.add_node("lead", self.lead_node)
        g.add_node("members", self.members_node)
        g.add_edge(START, "lead")
        g.add_edge("members", "lead")
        g.add_conditional_edges("lead", self.branch_node, ["members", END])
        return g.compile(checkpointer=self.memory)

    @property
    def lead_node(self):
        def lead(state: MessagesState):
            res = self.model.bind_tools(self.members).invoke(state["messages"])
            return {"messages": [res]}

        return lead

    @property
    def members_node(self):
        return MemberNode([*self.members])

    @property
    def designer_node(self):
        @member
        def designer(state: MessagesState) -> Command[Literal["lead"]]:
            """
            design the system and ensuring that it meets the requirements.
            """
            agent = Agent(self.model, [t.tavily(), t.duckduckgo(), t.serper()], self.memory, self.config)
            res = agent.invoke({"messages": state["messages"]})
            return {"messages": res["messages"]}

        return designer

    @property
    def programmer_node(self):
        @member
        def programmer(state: MessagesState) -> Command[Literal["lead"]]:
            """
            write the code and ensuring that it meets the requirements.
            """
            toolkit = [t.duckduckgo(), t.shell(), *t.file_management_tools()]
            agent = Agent(self.model, toolkit, self.memory, self.config)
            res = agent.invoke({"messages": state["messages"]})
            return {"messages": res["messages"]}

        return programmer

    @property
    def reviewer_node(self):
        @member
        def reviewer(state: MessagesState) -> Command[Literal["lead"]]:
            """
            review the code and ensuring that it meets the requirements.
            """
            toolkit = [t.duckduckgo(), *t.file_management_tools(), t.shell()]
            agent = Agent(self.model, toolkit, self.memory, self.config)
            res = agent.invoke({"messages": state["messages"]})
            return {"messages": res["messages"]}

        return reviewer

    @property
    def tester_node(self):
        @member
        def tester(state: MessagesState) -> Command[Literal["lead"]]:
            """
            test the code and ensuring that it meets the requirements.
            """
            toolkit = [t.tavily(), t.duckduckgo(), t.serper(), t.shell(), *t.file_management_tools(), t.python_repl()]
            agent = Agent(self.model, toolkit, self.memory, self.config)
            res = agent.invoke({"messages": state["messages"]})
            return {"messages": res["messages"]}

        return tester

    @property
    def branch_node(self):
        def should_continue(state: MessagesState):
            if state["messages"][-1].tool_calls:
                return "members"
            return END

        return should_continue

    async def start_interactive_chat(self) -> None:
        try:
            while True:
                user_input = input("user: ")
                if user_input == "q":
                    print("quitting...")
                    return
                print()
                async for i in self.graph.astream({"messages": [user_input]}, config=self.config, stream_mode="messages"):
                    print(i[0].content, end="")
                print("\n\n")
        except Exception as e:
            print(f"error: {e}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
