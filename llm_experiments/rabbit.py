from argparse import ArgumentParser, Namespace
from langchain.chat_models import init_chat_model
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage


class Agent:
    def __init__(self):
        model = init_chat_model("gpt-4o-mini", model_provider="openai")
        memory = MemorySaver()
        search = TavilySearchResults(max_results=2)
        tools = [search]
        self.agent = create_react_agent(model.bind_tools(tools), tools, checkpointer=memory)

    def ask(self, query: str):
        config = {"configurable": {"thread_id": "1"}}
        messages = [HumanMessage(content=query)]
        for step, metadata in self.agent.stream({"messages": messages}, config, stream_mode="messages"):
            if metadata["langgraph_node"] == "agent" and (text := step.text()):
                yield text


def parse_args() -> Namespace:
    parser = ArgumentParser()
    opt = parser.add_argument
    opt("--query", "-q", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    agent = Agent()
    for i in agent.ask(args.query):
        print(i, end="")