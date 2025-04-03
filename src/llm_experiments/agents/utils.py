from argparse import ArgumentParser

from langchain_core.messages import BaseMessage
from langgraph.graph import MessagesState


def parse_args():
    parser = ArgumentParser()
    opt = parser.add_argument
    opt("--model", "-m", type=str, default="4o-mini")
    opt("--verbose", "-v", action="store_true", default=False)
    return parser.parse_args()


def get_last_message(state: MessagesState) -> BaseMessage:
    return state["messages"][-1]


def stream_graph_updates(graph, user_input, config):
    for i in graph.stream({"messages": [user_input]}, config=config, stream_mode="updates"):
        print(f"agent: {i.get('agent').get('messages')[-1]}\n")
