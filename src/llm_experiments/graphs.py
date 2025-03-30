from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict, Annotated
from langchain.agents import create_react_agent
from langgraph.types import interrupt

from llm_experiments.llm import create_model
from llm_experiments import tools


class State(TypedDict):
    messages: Annotated[list[str], add_messages]


def researcher(state: State) -> dict[str, str]:
    model = create_model("gpt-o3-mini")
    agent = create_react_agent(
        model,
        [
            *tools.wikipedia(),
            *tools.serper(),
            *tools.tavily()
        ]
    )
    res = agent.invoke(state["messages"])
    return {"messages", [res["messages"][-1]]}


def human_node(state: State):
    user_input = interrupt("user: ")
    return {"messages": [user_input]}


def should_continue(state: State):
    if state["messages"][-1].content == "q":
        return END
    else:
        return "researcher"


def create_graph():
    graph = StateGraph(State)
    graph.add_node("researcher", researcher)
    graph.add_edge("human", human_node)
    graph.add_edge(START, "researcher")
    graph.add_conditional_edges("human", should_continue)
    graph.add_edge("researcher", END)
    return graph.compile()


def main():
    graph = create_graph()
    state = {
        "messages": [
            "You are a researcher. You will be asked to find information about a topic. "
            "You will be given a topic and you will have to find information about it. "
            "You can use Wikipedia, Serper and Tavily to find the information. "
            "You will be given a topic and you will have to find information about it."
        ]
    }
    result = graph.invoke(state)
    print(result)