from langgraph.prebuilt import create_react_agent

from llm_experiments import llm, tools


def todays_news(topic):
    model = llm.create_model("gpt-o3-mini")
    agent = create_react_agent(model, [*tools.tavily(), *tools.serper()])

    for i in agent.stream({"messages": f"search today's news about {topic}"}, stream_mode="values"):
        msg = i["messages"][-1]
        if isinstance(msg, tuple):
            print(msg)
        else:
            msg.pretty_print()


def research(prompt):
    model = llm.create_model("gpt-o3-mini")
    agent = create_react_agent(model, [*tools.tavily(), *tools.wikipedia()])

    for i in agent.stream({"messages": prompt}, stream_mode="values"):
        msg = i["messages"][-1]
        if isinstance(msg, tuple):
            print(msg)
        else:
            msg.pretty_print()
