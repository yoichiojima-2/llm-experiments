from langgraph.checkpoint.memory import MemorySaver

from llm_experiments import tools
from llm_experiments.agent import Agent
from llm_experiments.llm import create_model


def test_agent():
    config = {"configurable": {"thread_id": "test-agent"}}
    agent = Agent(model=create_model(), tools=tools.Tavily().get_tools(), memory=MemorySaver(), config=config)
    res = agent.invoke({"messages": ["search today's news"]}, config=config)
    print(res["messages"][-1].content)
    assert res
