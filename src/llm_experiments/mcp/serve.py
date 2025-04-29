from fastmcp import FastMCP

from llm_experiments import tools as t

mcp = FastMCP("llm-experiments")


@mcp.tool()
async def slack():
    toolkit = t.SlackToolkit()
    toolkit.get_tools()
