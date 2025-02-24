from argparse import ArgumentParser

from langchain_community.tools import ShellTool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langchain_core.messages.ai import AIMessageChunk
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from llm_experiments.models import instantiate_chat
from llm_experiments.vectorstore import VectorStore


class Agent:
    def __init__(self, chat_model="4o-mini", thread_id="default"):
        self.model = instantiate_chat(chat_model)
        self.config = {"configurable": {"thread_id": thread_id}}
        search = TavilySearchResults(max_results=2)
        shell = ShellTool()
        vs = VectorStore().tool
        self.tools = [search, shell, vs]
        self.tooled_model = self.model.bind_tools(self.tools)
        self.agent = create_react_agent(self.tooled_model, self.tools, checkpointer=MemorySaver())

    def stream(self, query):
        messages = [HumanMessage(content=query)]
        return self.agent.stream({"messages": messages}, self.config)

    def stream_messages(self, query):
        messages = [HumanMessage(content=query)]
        for i in self.agent.stream({"messages": messages}, self.config, stream_mode="messages"):
            if isinstance(i[0], AIMessageChunk):
                print(i[0].content, end="")

    def invoke(self, query):
        messages = [HumanMessage(content=query)]
        return self.agent.invoke({"messages": messages}, self.config)


def parse_args():
    parser = ArgumentParser()
    opt = parser.add_argument
    opt("--query", "-q", type=str, required=True)
    return parser.parse_args()


def main(query):
    agent = Agent()
    agent.stream_messages(query)


if __name__ == "__main__":
    args = parse_args()
    main(args.query)
