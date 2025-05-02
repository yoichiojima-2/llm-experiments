import sys
from fastmcp import FastMCP
from langchain_core.tools.base import BaseTool
from langgraph.checkpoint.memory import MemorySaver

from llm_experiments import prebuilt
from llm_experiments.llm import create_model


class MCPServer:
    def __init__(self, name: str, tools: list[BaseTool], tags: list[str] = []):
        self.mcp = FastMCP(name)
        self.tools = tools
        self.tags = [name, *tags]
        self.build()

    def build(self) -> FastMCP:
        for tool in self.tools:
            try:
                self.mcp.add_tool(tool.func, name=tool.name, description=tool.description, tags=self.tags)
            except Exception as e:
                print(file=sys.stderr)
                print(tool)

    def serve(self):
        self.mcp.run()


def main():
    swe = prebuilt.SWETeam(model=create_model(), memory=MemorySaver(), config={"configurable": {"thread_id": "mcp"}})
    swe = MCPServer(name="swe", tools=swe.tools)
    swe.serve()


if __name__ == "__main__":
    main()
