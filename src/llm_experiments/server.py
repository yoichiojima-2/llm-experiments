import tomllib
from pathlib import Path
from fastmcp import FastMCP


mcp = FastMCP("llm-experiments")


@mcp.resource("config://version")
def get_version() -> str:
    """get version from pyproject.toml"""
    with (Path(__file__).parent.parent.parent / "pyproject.toml").open("rb") as f:
        data = tomllib.load(f)
        return data["project"]["version"]


@mcp.tool()
def test_tool() -> str:
    """fastmcp tool test"""
    return "hello from tool"


@mcp.prompt()
def test_prompt() -> str:
    """fastmcp prompt test"""
    return "hello from prompt"


if __name__ == "__main__":
    mcp.run()