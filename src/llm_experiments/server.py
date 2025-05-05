from fastmcp import FastMCP

from llm_experiments import tools as t


class MCPServer:
    def __init__(self, name, tools: list[t.Tools] = None, tags=[]):
        self.server = FastMCP(name)
        self.name = name
        self.tools = tools
        self.tags = [name, *tags]
        self.build()

    def composite(self, server):
        self.server.mount(server.name, server.server)

    def build(self):
        if self.tools:
            for tools in self.tools:
                for tool in tools.get_tools():
                    self.server.add_tool(tool.func, name=tool.name, description=tool.description, tags=self.tags)

    def run(self):
        self.server.run()


def dev():
    return MCPServer(name="dev", tools=[t.Shell(), t.Python_()], tags=["dev"])


def research():
    return MCPServer(
        name="research",
        tools=[t.Wikipedia(), t.DuckDuckGo(), t.Tavily(), t.Serper()],
        tags=["research"],
    )


def slack():
    return MCPServer(name="slack", tools=[t.Slack()], tags=["slack"])


def main():
    main = MCPServer("llm_experiments")
    children = [
        dev(),
        research(),
        # slack()
    ]
    for child in children:
        main.composite(child)
    main.run()


if __name__ == "__main__":
    main()
