import sys
from abc import ABC
from dataclasses import dataclass
from pprint import pprint

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import ToolNode

from llm_experiments.tools import Tools


@dataclass
class AgentBase(ABC):
    model: BaseChatModel
    toolkits: list[Tools]
    memory: BaseCheckpointSaver
    graph: CompiledGraph

    def get_tools(self) -> list[BaseTool]:
        tools = []
        for toolkit in self.toolkits:
            for tool in toolkit.get_tools():
                tools.append(tool)
        return tools

    def invoke(self, *a, **kw):
        return self.graph.invoke(*a, **kw)

    async def interactive_chat(self, stream_mode="messages"):
        try:
            while True:
                user_input = input("user: ")
                if user_input == "q":
                    print("quitting...")
                    return
                print()
                async for i in self.graph.astream({"messages": [user_input]}, config=self.config, stream_mode=stream_mode):
                    match stream_mode:
                        case "messages":
                            print(i[0].content, end="")
                        case "debug":
                            pprint(i)
                        case _:
                            raise ValueError(f"unknown stream mode: {stream_mode}")
                print("\n\n")
        except Exception as e:
            print(f"error: {e}", file=sys.stderr)


class Agent(AgentBase):
    def __init__(self, model, toolkits, memory, config):
        self.model = model
        self.toolkits = toolkits
        self.memory = memory
        self.config = config
        self.graph = self.compile_graph()

    def compile_graph(self):
        g = StateGraph(MessagesState)
        g.add_node("agent", self.agent_node)
        g.add_node("tools", self.tools_node)
        g.add_edge(START, "agent")
        g.add_edge("tools", "agent")
        g.add_conditional_edges("agent", self.branch_node, ["tools", END])
        return g.compile(checkpointer=self.memory)

    @property
    def agent_node(self):
        def agent(state: MessagesState):
            res = self.model.bind_tools(self.get_tools()).invoke(state["messages"])
            return {"messages": [res]}

        return agent

    @property
    def tools_node(self):
        return ToolNode(self.get_tools())

    @property
    def branch_node(self):
        def should_continue(state: MessagesState):
            if state["messages"][-1].tool_calls:
                return "tools"
            return END

        return should_continue
