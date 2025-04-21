import tomllib
from pathlib import Path
from fastmcp import FastMCP


mcp = FastMCP("llm-experiments")


@mcp.tool()
def test_tool() -> str:
    return "hello from tool"


@mcp.resource("config://version")
def get_version() -> str:
    with (Path(__file__).parent.parent.parent / "pyproject.toml").open("rb") as f:
        data = tomllib.load(f)
        return data["project"]["version"]


@mcp.prompt()
def test_prompt() -> str:
    return "hello from prompt"


if __name__ == "__main__":
    mcp.run()