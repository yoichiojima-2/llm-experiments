import tomllib
from pathlib import Path
from fastmcp import Client


APP_ROOT = Path(__file__).parent.parent
SERVER_PATH = APP_ROOT / "src/llm_experiments/server.py"


async def test_server():
    async with Client(str(SERVER_PATH)) as client:
        await client.ping()

        tools = await client.list_tools()
        assert tools

        res = await client.call_tool("test_tool")
        assert res[0].text == "hello from tool"

        resources = await client.list_resources()
        assert resources

        with (APP_ROOT / "pyproject.toml").open("rb") as f:
            res = await client.read_resource("config://version")
            assert tomllib.load(f)["project"]["version"] == res[0].text

        prompts = await client.get_prompt("test_prompt")
        assert prompts