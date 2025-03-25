import asyncio
from typing import Annotated
from argparse import ArgumentParser

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from llm_experiments.llm import create_model


def parse_args():
    parser = ArgumentParser()
    opt = parser.add_argument
    opt("--model", default="gpt-4o-mini", help="model to use")
    opt("--thread-id", default="simple_chatbot", help="thread id")
    return parser.parse_args()


class State(TypedDict):
    messages: Annotated[list[str], add_messages]



class ChatBot:
    def __init__(self, model: str):
        self.model = create_model(model)

    def chat(self, state: State):
        return {"messages": [self.model.invoke(state["messages"])]}


def graph(model: str):
    chatbot = ChatBot(model)
    builder = StateGraph(State)
    builder.add_node("chatbot", chatbot.chat)
    builder.add_edge(START, "chatbot")
    builder.add_edge("chatbot", END)
    return builder.compile(checkpointer=MemorySaver())


async def main():
    args = parse_args()
    g = graph(args.model)
    config = {"thread_id": args.thread_id}
    while True:
        user_input = input("user: ")
        print("\n")
        print("assitant: ", end="")
        async for i in g.astream({"messages": [user_input]}, config=config, stream_mode="messages"):
            print(i[0].content, end="", flush=True)
        print("\n\n")


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
