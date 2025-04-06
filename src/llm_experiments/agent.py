import textwrap
from typing import Callable, Literal

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.tools.base import BaseTool
from langgraph.checkpoint.memory import BaseCheckpointSaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.types import Command

from llm_experiments import prompts

NodeType = Callable[[MessagesState], Command]


def create_executor(model, tools, verbose) -> AgentExecutor:
    prompt = prompts.multipurpose()
    agent = create_react_agent(model, tools, prompt=prompt)
    return AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True, verbose=verbose)


class Agent:
    def __init__(
        self,
        executor: AgentExecutor,
        model: BaseChatModel,
        memory: BaseCheckpointSaver,
        tools: list[BaseTool],
        verbose: bool = False,
        config: dict[dict[str, str]] = {"configurable": {"thread_id": "default"}},
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

        return superagent

    def create_node(self) -> NodeType:
        def node(state: MessagesState):
            res = self.executor.invoke({"input": state["messages"]})
            return {"messages": [res["output"]]}

        return node

    async def astream_messages(self, user_input: str) -> None:
        print()
        async for i in self.graph.astream({"messages": [user_input]}, config=self.config, stream_mode="messages"):
            print(i[0].content, end="")
        print("\n\n")

    async def interactive_chat(self) -> None:
        while True:
            user_input = input("user: ")
            if user_input == "q":
                print("quitting...")
                return
            await self.astream_messages(user_input)
