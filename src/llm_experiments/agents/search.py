from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_react_agent
from typing_extensions import TypedDict, Annotated
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from llm_experiments import prompts, tools
from llm_experiments.agents.utils import parse_args
from llm_experiments.llm import create_model


def agent(model, verbose=False) -> None:
    prompt = prompts.multipurpose()
    tool_list = [
        tools.duckduckgo(),
        tools.tavily(),
        tools.serper(),
        # tools.wikipedia(),
    ]
    agent = create_react_agent(model, tool_list, prompt=prompt)
    return AgentExecutor(agent=agent, tools=tool_list, handle_parsing_errors=True, verbose=verbose)


class State(TypedDict):
    messages: Annotated[str, add_messages]


def get_last_message(state: State) -> BaseMessage:
    return state["messages"][-1]


def get_search_node(model, verbose):
    def search_node(state: State):
        res = agent(model, verbose=verbose).invoke({"input": get_last_message(state)})
        return {"messages": [res["output"]]}
    return search_node


def graph(model, verbose):
    graph = StateGraph(State)
    graph.add_node("search", get_search_node(model, verbose))
    graph.add_edge(START, "search")
    graph.add_edge("search", END)
    return graph.compile(checkpointer=MemorySaver())


def main():
    load_dotenv()
    args = parse_args()
    model = create_model(args.model)
    config = {"configurable": {"thread_id": "search"}}
    for i in graph(model, verbose=args.verbose).stream({"messages": args.query}, config=config, stream_mode="updates"):
        print(i.get("search").get("messages")[0])


if __name__ == "__main__":
    main()
