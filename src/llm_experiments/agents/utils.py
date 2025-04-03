from argparse import ArgumentParser

from langchain_core.messages import BaseMessage
from langgraph.graph import MessagesState


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


def get_last_message(state: MessagesState) -> BaseMessage:
    return state["messages"][-1]


def stream_graph_updates(graph, user_input, config):
    for i in graph.stream({"messages": [user_input]}, config=config, stream_mode="messages"):
        print(i[0].content, end="")
