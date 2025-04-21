import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain.agents import create_react_agent

from llm_experiments.llm import create_model


async def main():
    model = create_model()

    server_params = StdioServerParameters(
        command="python",
        args=["-m", "llm_experiments.server"]
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            agent = create_react_agent(model=model, tools=tools)
            res = agent.ainvoke({"messages": "greet"})
            print(res)


if __name__ == "__main__":
    asyncio.run(main())
