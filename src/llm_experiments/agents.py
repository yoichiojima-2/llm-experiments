from langgraph.prebuilt import create_react_agent

from llm_experiments import llm, tools


def _pprint_stream(agent, prompt):
    for i in agent.stream({"messages": prompt}, stream_mode="values"):
        msg = i["messages"][-1]
        if isinstance(msg, tuple):
            print(msg)
        else:
            msg.pretty_print()


def todays_news(topic):
    model = llm.create_model("gpt-o3-mini")
    agent = create_react_agent(model, [*tools.tavily(), *tools.serper()])
    _pprint_stream(agent, f"search today's news about {topic}")


def research(prompt):
    model = llm.create_model("gpt-o3-mini")
    agent = create_react_agent(model, [*tools.tavily(), *tools.wikipedia()])
    _pprint_stream(agent, prompt)
