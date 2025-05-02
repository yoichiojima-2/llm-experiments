from langgraph.checkpoint.memory import MemorySaver

from llm_experiments import tools
from llm_experiments.agent import Agent
from llm_experiments.llm import create_model


def test_agent():
    model = create_model()
    toolkits = tools.Tavily()
    memory = MemorySaver()
    config = {"configurable": {"thread_id": "test-agent"}}

    agent = Agent(model=model, toolkits=[toolkits], memory=memory, config=config)
    res = agent.invoke({"messages": ["search today's news"]}, config=config)
    print(res["messages"][-1].content)
    assert res
