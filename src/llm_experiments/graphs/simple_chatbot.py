import asyncio
from typing import Annotated

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from llm_experiments.llm import create_model


class State(TypedDict):
    messages: Annotated[list[str], add_messages]


def chatbot(state: State):
    model = create_model()
    return {"messages": [model.invoke(state["messages"])]}


def graph():
    builder = StateGraph(State)
    builder.add_node("chatbot", chatbot)
    builder.add_edge(START, "chatbot")
    builder.add_edge("chatbot", END)
    return builder.compile(checkpointer=MemorySaver())


async def main(thread_id="simple_chatbot"):
    g = graph()
    config = {"thread_id": thread_id}
    while True:
        user_input = input("user: ")
        print("\n")
        print("assitant: ", end="")
        async for i in g.astream({"messages": [user_input]}, config=config, stream_mode="messages"):
            print(i[0].content, end="")
        print("\n\n")


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
