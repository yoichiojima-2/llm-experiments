import textwrap
from argparse import ArgumentParser
from typing import Literal

from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph import MessagesState
from langgraph.types import Command


def parse_args():
    parser = ArgumentParser()
    opt = parser.add_argument
    opt("--model", "-m", type=str, default="4o-mini")
    opt("--verbose", "-v", action="store_true", default=False)
    return parser.parse_args()


def create_node(agent):
    def node(state: MessagesState):
        res = agent.invoke({"input": state["messages"]})
        return {"messages": [res["output"]]}

    return node


def create_super_node(model, tools):
    def superagent(state: MessagesState) -> Command[Literal["agent", "__end__"]]:
        system_prompt = textwrap.dedent(
            f"""
            Answer the following questions as best you can
            You have access to the following tools:
            {tools}
            """
        )

        msgs = [SystemMessage(content=system_prompt), *state["messages"]]
        res = model.bind_tools(tools).invoke(msgs)

        if len(res.tool_calls) > 0:
            tool_call_id = res.tool_calls[-1]["id"]
            tool_msg = {
                "role": "tool",
                "content": "Successfully transferred",
                "tool_call_id": tool_call_id,
            }
            return Command(goto="agent", update={"messages": [res, tool_msg]})

    return superagent


def get_last_message(state: MessagesState) -> BaseMessage:
    return state["messages"][-1]


def stream_graph_updates(graph, user_input, config):
    for i in graph.stream({"messages": [user_input]}, config=config, stream_mode="messages"):
        print(i[0].content, end="")

    print("\n\n")
