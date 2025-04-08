import sys

from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode


class Agent:
    def __init__(self, model, tools, memory, config):
        self.model = model
        self.tools = tools
        self.memory = memory
        self.config = config
        self.graph = self.compile_graph()

    def create_agent(self):
        def agent(state: MessagesState):
            res = self.model.bind_tools(self.tools).invoke(state["messages"])
            return {"messages": [res]}

        return agent

    def create_tools(self):
        return ToolNode(self.tools)

    def create_should_continue(self):
        def should_continue(state: MessagesState):
            if state["messages"][-1].tool_calls:
                return "tools"
            return END

        return should_continue

    def compile_graph(self):
        g = StateGraph(MessagesState)
        g.add_node("agent", self.create_agent())
        g.add_node("tools", self.create_tools())
        g.add_edge(START, "agent")
        g.add_edge("tools", "agent")
        g.add_conditional_edges("agent", self.create_should_continue(), ["tools", END])
        return g.compile(checkpointer=self.memory)

    async def start_interactive_chat(self) -> None:
        try:
            while True:
                user_input = input("user: ")
                if user_input == "q":
                    print("quitting...")
                    return
                print()
                async for i in self.graph.astream({"messages": [user_input]}, config=self.config, stream_mode="messages"):
                    print(i[0].content, end="")
                print("\n\n")
        except Exception as e:
            print(f"error: {e}", file=sys.stderr)
