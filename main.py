import asyncio
from argparse import ArgumentParser

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

import nodes

load_dotenv()


def parse_args():
    parser = ArgumentParser()
    opt = parser.add_argument
    opt("-q", "--query", required=True)
    return parser.parse_args()


async def run(query, thread_id="1"):
    model = init_chat_model("gpt-4o-mini", model_provider="openai")
    config = {"configurable": {"thread_id": thread_id}}

    memory = MemorySaver()
    supervisor = nodes.SupervisorNode(model, checkpointer=memory).node()
    spotify = nodes.SpotifyNode(model, checkpointer=memory).node()
    shell = nodes.ShellNode(model, checkpointer=memory).node()

    graph = StateGraph(MessagesState)
    graph.add_node("supervisor", supervisor)
    graph.add_node("spotify", spotify)
    graph.add_node("shell", shell)
    graph.add_edge(START, "supervisor")
    app = graph.compile()

    res = await app.ainvoke({"messages": [query]}, config)
    print(res)


async def main():
    args = parse_args()
    await run(args.query)


if __name__ == "__main__":
    asyncio.run(main())
