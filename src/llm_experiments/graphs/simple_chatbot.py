from typing import Annotated

from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class State(TypedDict):
    messages: Annotated[list[str], add_messages]


def model():
    return init_chat_model("gpt-4o-mini", model_provider="openai")


def chatbot(state: State):
    return {"messages": [model().invoke(state["messages"])]}


def graph():
    graph_builder = StateGraph(State)
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)
    return graph_builder.compile(checkpointer=MemorySaver())


def main():
    g = graph()
    config = {"thread_id": "test"}
    while True:
        user_input = input("user: ")
        res = g.invoke({"messages": [user_input]}, config=config)
        print(f"assistant: {res['messages'][-1].content}")


main()
