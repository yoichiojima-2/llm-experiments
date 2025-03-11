import asyncio
from argparse import ArgumentParser

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

from langgraph.pregel import RetryPolicy
import agents
from nodes import Node, SupervisorNode, SpotifyNode
from utils import Playwright, print_stream

load_dotenv()


def parse_args():
    parser = ArgumentParser()
    opt = parser.add_argument
    opt("-q", "--query", required=True)
    return parser.parse_args()


def graph(model, playwright):
    memory = MemorySaver()

    supervisor = SupervisorNode(model, checkpointer=memory).node()
    spotify = SpotifyNode(model, checkpointer=memory).node()
    shell = Node.new(model, agents.ShellAgent, checkpointer=memory).node()
    python = Node.new(model, agents.PythonAgent, checkpointer=memory).node()
    duckduckgo = Node.new(model, agents.DuckDuckGoAgent, checkpointer=memory).node()
    wikipedia = Node.new(model, agents.WikipediaAgent, checkpointer=memory).node()
    browser = Node.new(model, agents.BrowserAgent, playwright, checkpointer=memory).node()
    files = Node.new(model, agents.FileAgent, checkpointer=memory).node()
    serper = Node.new(model, agents.SerperAgent, checkpointer=memory).node()
    tavily = Node.new(model, agents.TavilyAgent, checkpointer=memory).node()
    sql = Node.new(model, agents.SQLAgent, "test.db", checkpointer=memory).node()

    graph = StateGraph(MessagesState)
    graph.add_node("supervisor", supervisor)
    graph.add_node("spotify", spotify, retry=RetryPolicy())
    graph.add_node("shell", shell)
    graph.add_node("python", python)
    graph.add_node("duckduckgo", duckduckgo)
    graph.add_node("wikipedia", wikipedia)
    graph.add_node("browser", browser)
    graph.add_node("files", files)
    graph.add_node("serper", serper)
    graph.add_node("tavily", tavily)
    graph.add_node("sql", sql)
    graph.add_edge(START, "supervisor")

    return graph.compile()


async def run(query, thread_id="1"):
    model = init_chat_model("gpt-4o-mini", model_provider="openai")
    config = {"configurable": {"thread_id": thread_id}}
    async with Playwright() as playwright:
        app = graph(model, playwright)
        res = app.astream({"messages": [query]}, config, stream_mode="values")
        await print_stream(res)


async def main():
    args = parse_args()
    await run(args.query)


if __name__ == "__main__":
    asyncio.run(main())
