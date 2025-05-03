from fastmcp import FastMCP
from langchain_core.tools.base import BaseTool
from langgraph.checkpoint.memory import MemorySaver

from llm_experiments import prebuilt
from llm_experiments import tools as t
from llm_experiments.llm import create_model


class MCPServer:
    def __init__(self, name: str, tools: list[BaseTool] = [], tags: list[str] = []):
        self.mcp = FastMCP(name)
        self.name = name
        self.tools = tools
        self.tags = [name, *tags]
        self.build()

    def build(self) -> FastMCP:
        for tool in self.tools:
            self.mcp.add_tool(tool.func, name=tool.name, description=tool.description, tags=self.tags)

    def serve(self):
        self.mcp.run()

    def mount(self, server: FastMCP):
        self.mcp.mount(server.name, server.mcp)


def build_swe() -> MCPServer:
    swe_agent = prebuilt.SWETeam(model=create_model(), memory=MemorySaver(), config={"configurable": {"thread_id": "mcp"}})
    return MCPServer(name="swe", tools=swe_agent.tools)


def build_dev() -> MCPServer:
    return MCPServer(name="dev", tools=[*t.Shell().get_tools(), *t.Python_().get_tools()], tags=["dev"])


def build_research() -> MCPServer:
    return MCPServer(
        name="research",
        tools=[
            *t.Wikipedia().get_tools(),
            *t.DuckDuckGo().get_tools(),
            *t.Tavily().get_tools(),
            *t.Serper().get_tools(),
        ],
        tags=["research"],
    )


def build_slack() -> MCPServer:
    return MCPServer(name="slack", tools=[*t.Slack().get_tools()], tags=["slack"])


# fixme
# def build_sql(llm: BaseChatModel, db_name: str) -> MCPServer:
#     return MCPServer(name="sql", tools=t.SQL(llm=llm, db_name=db_name).get_tools(), tags=["sql"])


def main():
    main = MCPServer("llm_experiments")

    children = [build_swe(), build_dev(), build_research(), build_slack()]
    for child in children:
        main.mount(child)

    main.serve()


if __name__ == "__main__":
    main()
