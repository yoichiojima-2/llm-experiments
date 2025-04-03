from langgraph.prebuilt import create_react_agent

from llm_experiments import tools
from llm_experiments.llm import create_model


def test_duckduckgo_tools():
    agent = create_react_agent(create_model(), [tools.duckduckgo()])
    res = agent.invoke({"messages": "search today's news"})
    last_msg = res["messages"][-1].content
    assert last_msg
    print(last_msg)


def test_browser(): ...


def test_shell_tools():
    agent = create_react_agent(create_model(), [tools.shell()])
    res = agent.invoke({"messages": "ls"})
    for msg in res["messages"]:
        print(msg.content)
    assert True


def test_python_repl_tools():
    agent = create_react_agent(create_model(), [tools.python_repl()])
    res = agent.invoke({"messages": "print('hello world')"})
    for msg in res["messages"]:
        print(msg.content)
    assert True


def test_wikipedia_tools():
    agent = create_react_agent(create_model(), [tools.wikipedia()])
    res = agent.invoke({"messages": "search for monty python"})
    for msg in res["messages"]:
        print(msg.content)
    assert True


def test_file_management_tools():
    agent = create_react_agent(create_model(), tools.file_management_tools())
    res = agent.invoke({"messages": "list files in current directory"})
    for msg in res["messages"]:
        print(msg.content)
    assert True


def test_serper_tools():
    agent = create_react_agent(create_model(), [tools.serper()])
    res = agent.invoke({"messages": "search for today's news"})
    for msg in res["messages"]:
        print(msg.content)
    assert True


def test_tavily_tools():
    agent = create_react_agent(create_model(), [tools.tavily()])
    res = agent.invoke({"messages": "search for today's news"})
    for msg in res["messages"]:
        print(msg.content)
    assert True


def test_sql_tools():
    model = create_model()
    agent = create_react_agent(model, tools.sql_tools(model, "test"))
    res = agent.invoke({"messages": "select * from test"})
    for msg in res["messages"]:
        print(msg.content)
    assert True
