from fastmcp import FastMCP

from llm_experiments import tools as t


def main():
    tools = [t.DuckDuckGo(), t.Shell(), t.Python_(), t.Wikipedia(), t.Serper(), t.Slack()]
    tags = ["llm_experiments"]
    server = MCPServer(tools=tools, tags=tags)
    server.serve()


class MCPServer:
    def __init__(self, tools: list[t.Tools], tags: list[str] = None):
        self.mcp = FastMCP("llm_experiments")
        self.tools = tools
        self.tags = tags
        self.build()

    def build(self) -> FastMCP:
        for tools in self.tools:
            for tool in tools.get_tools():
                self.mcp.add_tool(tool.func, name=tool.name, description=tool.description, tags=self.tags)

    def serve(self):
        self.mcp.run()


if __name__ == "__main__":
    main()
