import asyncio
from typing import Annotated
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


load_dotenv()

class State(TypedDict):
    messages: Annotated[list[str], add_messages]


def chatbot(state: State):
    model = init_chat_model("gpt-4o-mini", model_provider="openai")
    return {"messages": [model.invoke(state["messages"])]}


def graph():
    builder = StateGraph(State)
    builder.add_node("chatbot", chatbot)
    builder.add_edge(START, "chatbot")
    builder.add_edge("chatbot", END)
    return builder.compile(checkpointer=MemorySaver())


async def main():
    g = graph()
    config = {"thread_id": "test"}
    while True:
        user_input = input("user: ")
        res = await g.ainvoke({"messages": [user_input]}, config=config)
        print(f"assistant: {res['messages'][-1].content}")


asyncio.run(main())
