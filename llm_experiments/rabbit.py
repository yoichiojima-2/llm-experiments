import typing as t
from argparse import ArgumentParser, Namespace

from langchain_community.tools import ShellTool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from llm_experiments.chat import instantiate_chat


class Agent:
    def __init__(self, config: dict[str, t.Any]):
        self.config = config
        model = instantiate_chat("4o-mini")
        memory = MemorySaver()
        search = TavilySearchResults(max_results=2)
        shell = ShellTool()
        tools = [search, shell]
        self.agent = create_react_agent(
            model.bind_tools(tools), tools, checkpointer=memory
        )

    def stream(self, query: str):
        messages = [HumanMessage(content=query)]
        for step, metadata in self.agent.stream(
            {"messages": messages}, self.config, stream_mode="messages"
        ):
            if metadata["langgraph_node"] == "agent" and (text := step.text()):
                yield text

    def invoke(self, query: str):
        messages = [HumanMessage(content=query)]
        for step in self.agent.invoke(
            {"messages": messages}, self.config, stream_mode="values"
        ):
            return step


def parse_args() -> Namespace:
    parser = ArgumentParser()
    opt = parser.add_argument
    opt("--query", "-q", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = {"configurable": {"thread_id": "rabbit"}}
    agent = Agent(config=config)
    for i in agent.stream(args.query):
        print(i, end="")
    print()
