import asyncio
from argparse import ArgumentParser

import agents
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.pregel import RetryPolicy
from nodes import Node, SpotifyNode, SupervisorNode, UserNode
from utils import Playwright

load_dotenv()


def parse_args():
    parser = ArgumentParser()
    opt = parser.add_argument
    opt("-q", "--query", required=True)
    opt("--thread-id", default="default")
    return parser.parse_args()


def graph(model, db_name, playwright):
    supervisor = SupervisorNode(model).node()
    spotify = SpotifyNode(model).node()
    user = UserNode(model).node()
    shell = Node.new(model, agents.ShellAgent).node()
    python = Node.new(model, agents.PythonAgent).node()
    duckduckgo = Node.new(model, agents.DuckDuckGoAgent).node()
    wikipedia = Node.new(model, agents.WikipediaAgent).node()
    browser = Node.new(model, agents.BrowserAgent, playwright).node()
    files = Node.new(model, agents.FileAgent).node()
    serper = Node.new(model, agents.SerperAgent).node()
    tavily = Node.new(model, agents.TavilyAgent).node()
    sql = Node.new(model, agents.SQLAgent, db_name).node()

    g = StateGraph(MessagesState)
    g.add_node("supervisor", supervisor)
    g.add_node("spotify", spotify, retry=RetryPolicy())
    g.add_node("shell", shell)
    g.add_node("python", python)
    g.add_node("duckduckgo", duckduckgo)
    g.add_node("wikipedia", wikipedia)
    g.add_node("browser", browser)
    g.add_node("files", files)
    g.add_node("serper", serper)
    g.add_node("tavily", tavily)
    g.add_node("sql", sql)
    g.add_node("user", user)
    g.add_edge(START, "supervisor")
    return g.compile(checkpointer=MemorySaver())


async def run(query, thread_id="1"):
    model = init_chat_model("gpt-4o-mini", model_provider="openai")
    config = {"configurable": {"thread_id": thread_id}}

    with Playwright() as pw:
        app = graph(model, "test.db", pw)
        async for res in app.astream({"messages": [query]}, config, stream_mode="update"):
            print(res)


async def main():
    args = parse_args()
    await run(args.query, args.thread_id)


if __name__ == "__main__":
    asyncio.run(main())
