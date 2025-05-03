import nest_asyncio
from langchain_community.tools.playwright.utils import create_async_playwright_browser
from langgraph.prebuilt import create_react_agent

from llm_experiments import tools
from llm_experiments.llm import create_model

nest_asyncio.apply()


def test_slack():
    sl = tools.Slack()
    res = sl.get_tools_by_name()["post_message"].invoke({"channel": "test", "text": "hello from agent"})
    assert res["ok"]
    res = sl.get_tools_by_name()["delete_message"].invoke({"channel": res["channel"], "ts": res["ts"]})
    assert res["ok"]


def test_duckduckgo():
    agent = create_react_agent(create_model(), tools.DuckDuckGo().get_tools())
    res = agent.invoke({"messages": "search today's news"})
    last_msg = res["messages"][-1].content
    assert last_msg
    print(last_msg)


async def test_browser():
    async with create_async_playwright_browser() as browser:
        toolkit = tools.Browser(browser=browser).get_tools()
        agent = create_react_agent(create_model(), toolkit)
        res = agent.invoke({"messages": "go to wikipedia.org"})
        last_msg = res["messages"][-1].content
        assert last_msg
        print(last_msg)


def test_shell():
    agent = create_react_agent(create_model(), tools.Shell().get_tools())
    res = agent.invoke({"messages": "ls"})
    for msg in res["messages"]:
        print(msg.content)
    assert True


def test_python():
    agent = create_react_agent(create_model(), tools.Python_().get_tools())
    res = agent.invoke({"messages": "print('hello world')"})
    for msg in res["messages"]:
        print(msg.content)
    assert True


def test_wikipedia():
    agent = create_react_agent(create_model(), tools.Wikipedia().get_tools())
    res = agent.invoke({"messages": "search for monty python"})
    for msg in res["messages"]:
        print(msg.content)
    assert True


def test_serper():
    agent = create_react_agent(create_model(), tools.Serper().get_tools())
    res = agent.invoke({"messages": "search for today's news"})
    for msg in res["messages"]:
        print(msg.content)
    assert True


def test_tavily():
    agent = create_react_agent(create_model(), tools.Tavily().get_tools())
    res = agent.invoke({"messages": "search for today's news"})
    for msg in res["messages"]:
        print(msg.content)
    assert True


def test_sql():
    model = create_model()
    agent = create_react_agent(model, tools.SQL(llm=model, db_name="test").get_tools())
    res = agent.invoke({"messages": "select * from test"})
    for msg in res["messages"]:
        print(msg.content)
    assert True
