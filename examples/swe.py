"""
this example demonstrates how to create a software engineering team
"""

import asyncio
import sys
from argparse import ArgumentParser
from pathlib import Path
from pprint import pprint

from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from llm_experiments import tools as t
from llm_experiments.agent import Agent
from llm_experiments.llm import create_model


def parse_args():
    parser = ArgumentParser(description="SWE team example")
    parser.add_argument("--workdir", "-w", type=str, default="output/swe")
    parser.add_argument("--model", "-m", type=str, default="o3-mini")
    parser.add_argument("--thread_id", "-t", type=str, default="swe")
    parser.add_argument("--stream-mode", "-s", type=str, default="messages")
    return parser.parse_args()


async def main():
    args = parse_args()
    config = {"configurable": {"thread_id": args.thread_id}}
    dev_team = SWE_Team(
        create_model(args.model),
        MemorySaver(),
        config,
        workdir=args.workdir,
    )
    await dev_team.start_interactive_chat(stream_mode=args.stream_mode)  # todo: revert to "messages" for production)


class SWE_Team:
    def __init__(self, model, memory, config, workdir="output/swe"):
        self.model = model
        self.memory = memory
        self.config = config

        self.workdir = Path(workdir)
        if not self.workdir.exists():
            self.workdir.mkdir(parents=True, exist_ok=True)

        self.tools = [
            self.designer_node,
            self.programmer_node,
            self.reviewer_node,
            self.tester_node,
            *t.file_management_tools(root_dir=str(workdir)),
            t.shell(),
        ]

        self.graph = self.compile_graph()

    def compile_graph(self):
        g = StateGraph(MessagesState)
        g.add_node("lead", self.lead_node)
        g.add_node("members", ToolNode([*self.tools]))
        g.add_edge(START, "lead")
        g.add_edge("members", "lead")
        g.add_conditional_edges("lead", self.branch_node, ["members", END])
        return g.compile(checkpointer=self.memory)

    @property
    def lead_node(self):
        def lead(state: MessagesState):
            payload = [
                SystemMessage(
                    content=(
                        "You are a software engineering team lead. "
                        "You will be responsible for leading the team and ensuring that the system meets the requirements. "
                        "lead members specialized in design, programming, reviewing, and testing. "
                    )
                ),
                *state["messages"],
            ]
            res = self.model.bind_tools(self.tools).invoke(payload)
            return {"messages": [res]}

        return lead

    @property
    def designer_node(self):
        @tool
        def designer(state: MessagesState):
            """
            design the system and ensuring that it meets the requirements.
            """
            agent = Agent(
                self.model,
                [t.tavily(), t.duckduckgo(), t.serper(), *t.file_management_tools(root_dir=str(self.workdir))],
                self.memory,
                self.config,
            )
            res = agent.invoke({"messages": state["messages"]})
            return {"messages": res["messages"]}

        return designer

    @property
    def programmer_node(self):
        @tool
        def programmer(state: MessagesState):
            """
            write the code and ensuring that it meets the requirements.
            """
            toolkit = [t.duckduckgo(), t.shell(), *t.file_management_tools(root_dir=str(self.workdir))]
            agent = Agent(self.model, toolkit, self.memory, self.config)
            res = agent.invoke({"messages": state["messages"]})
            return {"messages": res["messages"]}

        return programmer

    @property
    def reviewer_node(self):
        @tool
        def reviewer(state: MessagesState):
            """
            review the code and ensuring that it meets the requirements.
            """
            toolkit = [t.duckduckgo(), *t.file_management_tools(root_dir=str(self.workdir)), t.shell()]
            agent = Agent(self.model, toolkit, self.memory, self.config)
            res = agent.invoke({"messages": state["messages"]})
            return {"messages": res["messages"]}

        return reviewer

    @property
    def tester_node(self):
        @tool
        def tester(state: MessagesState):
            """
            test the code and ensuring that it meets the requirements.
            """
            toolkit = [
                t.tavily(),
                t.duckduckgo(),
                t.serper(),
                t.shell(),
                *t.file_management_tools(root_dir=str(self.workdir)),
                t.python_repl(),
            ]
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

    async def start_interactive_chat(self, stream_mode="messages") -> None:
        try:
            while True:
                user_input = input("user: ")
                if user_input == "q":
                    print("quitting...")
                    return
                print()
                async for i in self.graph.astream({"messages": [user_input]}, config=self.config, stream_mode=stream_mode):
                    match stream_mode:
                        case "messages":
                            print(i[0].content, end="")
                        case "debug":
                            pprint(i)
                        case "_":
                            raise ValueError(f"unknown stream mode: {stream_mode}")

                print("\n\n")
        except Exception as e:
            print(f"error: {e}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
