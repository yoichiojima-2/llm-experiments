from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

from llm_experiments import prompts, tools
from llm_experiments.agents.utils import create_node, parse_args, stream_graph_updates
from llm_experiments.llm import create_model


def create_agent(model, verbose=False) -> None:
    prompt = prompts.multipurpose()
    tool_list = [tools.shell()]
    agent = create_react_agent(model, tool_list, prompt=prompt)
    return AgentExecutor(agent=agent, tools=tool_list, handle_parsing_errors=True, verbose=verbose)


def create_graph(model, verbose):
    agent = create_agent(model, verbose=verbose)
    graph = StateGraph(MessagesState)
    graph.add_node("agent", create_node(agent))
    graph.add_edge(START, "agent")
    graph.add_edge("agent", END)
    return graph.compile(checkpointer=MemorySaver())


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
