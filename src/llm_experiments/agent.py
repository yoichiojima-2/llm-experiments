import textwrap
from typing import Callable, Literal

from langchain.agents import AgentExecutor
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.tools.base import BaseTool
from langgraph.checkpoint.memory import BaseCheckpointSaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.types import Command

NodeType = Callable[[MessagesState], Command]


class Agent:
    def __init__(
        self,
        executor: AgentExecutor,
        model: BaseChatModel,
        memory: BaseCheckpointSaver,
        tools: list[BaseTool],
        verbose: bool = False,
        config: dict[str, dict[str, str]] = {"configurable": {"thread_id": "default"}},
    ):
        self.executor = executor
        self.model = model
        self.memory = memory
        self.verbose = verbose
        self.tools = tools
        self.config = config
        self.graph = self.compile_graph()

    def compile_graph(self) -> CompiledGraph:
        g = StateGraph(MessagesState)
        g.add_node("superagent", self.create_super_node())
        g.add_node("agent", self.create_node())
        g.add_edge(START, "superagent")
        g.add_edge("superagent", "agent")
        return g.compile(checkpointer=self.memory)

    def create_super_node(self) -> NodeType:
        def superagent(state: MessagesState) -> Command[Literal["agent", "__end__"]]:
            system_prompt = textwrap.dedent(
                f"""
                Answer the following questions as best you can.
                You have access to the following tools:
                {self.tools}
                If you don't need agent with tools, send to agent_w_no_tool.
                """
            )
            msgs = [SystemMessage(content=system_prompt), *state["messages"]]
            res = self.model.bind_tools(self.tools).invoke(msgs)
            if len(res.tool_calls) > 0:
                tool_call_id = res.tool_calls[-1]["id"]
                tool_msg = {
                    "role": "tool",
                    "content": "Successfully transferred",
                    "tool_call_id": tool_call_id,
                }
                return Command(goto="agent", update={"messages": [res, tool_msg]})
            else:
                return Command(goto="__end__", update={"messages": [res]})

        return superagent

    def create_node(self) -> NodeType:
        def node(state: MessagesState):
            res = self.executor.invoke({"input": state["messages"]})
            return {"messages": [res["output"]]}

        return node
