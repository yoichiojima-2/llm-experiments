import pytest
from langchain.chat_models import init_chat_model

from llm_experiments.agents import (
    BrowserAgent,
    DuckDuckGoAgent,
    FileAgent,
    PythonAgent,
    SerperAgent,
    ShellAgent,
    SpotifyAgent,
    SQLAgent,
    TavilyAgent,
    WikipediaAgent,
)


@pytest.fixture
def model():
    return init_chat_model("gpt-4o-mini", model_provider="openai")


@pytest.mark.asyncio
async def test_spotify_agent(model):
    agent = SpotifyAgent(model)
    res = await agent.agent().ainvoke({"input": "show me the tracklist of sgt. pepper's lonely hearts club band"})
    print(agent.get_last_response(res))
    assert True


@pytest.mark.asyncio
async def test_duck_duck_go_agent(model):
    agent = DuckDuckGoAgent(model)
    res = await agent.agent().ainvoke({"messages": "what is the capital of france?"})
    print(agent.get_last_response(res))
    assert True


@pytest.mark.asyncio
async def test_shell_agent(model):
    agent = ShellAgent(model)
    res = await agent.agent().ainvoke({"messages": "ls"})
    print(agent.get_last_response(res))
    assert True


@pytest.mark.asyncio
async def test_browser_agent(model):
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        agent = BrowserAgent(model, p)
        agt = await agent.agent(verbose=True)
        res = await agt.ainvoke({"input": "search today's news"})
        print(agent.get_last_response(res))
        assert True


@pytest.mark.asyncio
async def test_python_agent(model):
    agent = PythonAgent(model)
    res = await agent.agent().ainvoke({"messages": "print('hello world')"})
    print(agent.get_last_response(res))
    assert True


@pytest.mark.asyncio
async def test_wikipedia_agent(model):
    agent = WikipediaAgent(model)
    res = await agent.agent().ainvoke({"messages": "who is the president of the united states?"})
    print(agent.get_last_response(res))
    assert True


@pytest.mark.asyncio
async def test_file_agent(model):
    agent = FileAgent(model)
    res = await agent.agent().ainvoke({"messages": "ls"})
    print(agent.get_last_response(res))
    assert True


@pytest.mark.asyncio
async def test_serper_agent(model):
    agent = SerperAgent(model)
    res = await agent.agent().ainvoke({"messages": "what is the capital of france?"})
    print(agent.get_last_response(res))
    assert True


@pytest.mark.asyncio
async def test_tavily_agent(model):
    agent = TavilyAgent(model)
    res = await agent.agent().ainvoke({"messages": "what is the capital of france?"})
    print(agent.get_last_response(res))
    assert True


@pytest.mark.asyncio
async def test_sql_agent(model):
    agent = SQLAgent(model, db_name="test")
    res = await agent.agent().ainvoke({"messages": "select * from users"})
    print(agent.get_last_response(res))
    assert True
