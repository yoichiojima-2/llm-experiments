from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

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


def get_last_message(state: MessagesState) -> BaseMessage:
    return state["messages"][-1]


def get_search_node(model, verbose):
    def search_node(state: MessagesState):
        res = agent(model, verbose=verbose).invoke({"input": get_last_message(state)})
        return {"messages": [res["output"]]}

    return search_node


def create_graph(model, verbose):
    graph = StateGraph(MessagesState)
    graph.add_node("agent", get_search_node(model, verbose))
    graph.add_edge(START, "agent")
    graph.add_edge("agent", END)
    return graph.compile(checkpointer=MemorySaver())


def stream_graph_updates(graph, user_input, config):
    for i in graph.stream({"messages": [user_input]}, config=config, stream_mode="updates"):
        print(f"agent: {i.get('agent').get('messages')[-1]}\n")


def main():
    load_dotenv()
    args = parse_args()
    model = create_model(args.model)
    graph = create_graph(model, verbose=args.verbose)
    config = {"configurable": {"thread_id": "search"}}
    while True:
        user_input = input("user: ")
        if user_input == "q":
            break
        stream_graph_updates(graph, user_input, config)


if __name__ == "__main__":
    main()
