from fastmcp import FastMCP


mcp = FastMCP("llm-experiments")

@mcp.tool()
def greet() -> str:
    return "hello"


if __name__ == "__main__":
    mcp.run()