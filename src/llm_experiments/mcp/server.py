import tomllib
from pathlib import Path

from fastmcp import FastMCP

mcp = FastMCP("llm-experiments")


if __name__ == "__main__":
    mcp.run()
