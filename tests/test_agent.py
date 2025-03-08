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
    agent = SpotifyAgent().agent(model)
    await agent.ainvoke({"input": "what is the most popular song on spotify?"})
    assert True


@pytest.mark.asyncio
async def test_duck_duck_go_agent(model):
    agent = DuckDuckGoAgent().agent(model)
    await agent.ainvoke({"messages": "what is the capital of france?"})
    assert True


@pytest.mark.asyncio
async def test_shell_agent(model):
    agent = ShellAgent().agent(model)
    await agent.ainvoke({"messages": "ls"})
    assert True


@pytest.mark.asyncio
async def test_browser_agent(model):
    agent = await BrowserAgent().agent(model)
    await agent.ainvoke({"messages": "what is the capital of france?"})
    assert True


@pytest.mark.asyncio
async def test_python_agent(model):
    agent = PythonAgent().agent(model)
    await agent.ainvoke({"messages": "print('hello world')"})
    assert True


@pytest.mark.asyncio
async def test_wikipedia_agent(model):
    agent = WikipediaAgent().agent(model)
    await agent.ainvoke({"messages": "who is the president of the united states?"})
    assert True


@pytest.mark.asyncio
async def test_file_agent(model):
    agent = FileAgent().agent(model)
    await agent.ainvoke({"messages": "ls"})
    assert True


@pytest.mark.asyncio
async def test_serper_agent(model):
    agent = SerperAgent().agent(model)
    await agent.ainvoke({"messages": "what is the capital of france?"})
    assert True


@pytest.mark.asyncio
async def test_tavily_agent(model):
    agent = TavilyAgent().agent(model)
    await agent.ainvoke({"messages": "what is the capital of france?"})
    assert True


@pytest.mark.asyncio
async def test_sql_agent(model):
    agent = SQLAgent().agent(model, db_name="test")
    await agent.ainvoke({"messages": "select * from users"})
    assert True
