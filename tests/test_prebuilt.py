import nest_asyncio
import pytest
from langchain_community.tools.playwright.utils import create_async_playwright_browser
from langgraph.checkpoint.memory import MemorySaver

from llm_experiments import prebuilt
from llm_experiments.llm import create_model

nest_asyncio.apply()


@pytest.fixture
def model():
    return create_model("4o-mini")


@pytest.fixture
def memory():
    return MemorySaver()


@pytest.fixture
def config():
    return {"configurable": {"thread_id": "test-prebuilt"}}


async def test_swe(model, memory, config):
    team = await prebuilt.swe(model, memory, config, workdir="output/swe")
    res = team.invoke({"messages": "this is a test"}, config=config)
    print(res["messages"][-1].content)
    assert res


async def test_search(model, memory, config):
    agent = await prebuilt.search(model, memory, config)
    res = agent.invoke({"messages": "this is a test"}, config=config)
    print(res["messages"][-1].content)
    assert res


async def test_shell(model, memory, config):
    agent = await prebuilt.shell(model, memory, config)
    res = agent.invoke({"messages": "this is a test"}, config=config)
    print(res["messages"][-1].content)
    assert res


async def test_shell_w_search(model, memory, config):
    agent = await prebuilt.shell_w_search(model, memory, config)
    res = agent.invoke({"messages": "this is a test"}, config=config)
    print(res["messages"][-1].content)
    assert res


async def test_slack(model, memory, config):
    agent = await prebuilt.slack(model, memory, config)
    res = agent.invoke({"messages": "this is a test"}, config=config)
    print(res["messages"][-1].content)
    assert res


async def test_python_repl(model, memory, config):
    agent = await prebuilt.python_repl(model, memory, config)
    res = agent.invoke({"messages": "this is a test"}, config=config)
    print(res["messages"][-1].content)
    assert res


async def test_browser(model, memory, config):
    async with create_async_playwright_browser() as b:
        agent = await prebuilt.browser(model, memory, config, b)
        res = agent.invoke({"messages": "this is a test"}, config=config)
        print(res["messages"][-1].content)
        assert res


async def test_browser_w_search(model, memory, config):
    async with create_async_playwright_browser() as b:
        agent = await prebuilt.browser_w_search(model, memory, config, b)
        res = agent.invoke({"messages": "this is a test"}, config=config)
        print(res["messages"][-1].content)
        assert res
