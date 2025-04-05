from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

from llm_experiments import prompts, tools
from llm_experiments.graphs.utils import create_node, create_super_node, parse_args, stream_graph_updates
from llm_experiments.llm import create_model

TOOLS = [tools.duckduckgo(), tools.tavily(), tools.serper()]


def create_agent(model, verbose=False) -> None:
    prompt = prompts.multipurpose()
    agent = create_react_agent(model, TOOLS, prompt=prompt)
    return AgentExecutor(agent=agent, tools=TOOLS, handle_parsing_errors=True, verbose=verbose)


def create_graph(model, verbose):
    graph = StateGraph(MessagesState)
    agent = create_agent(model, verbose=verbose)
    graph.add_node("superagent", create_super_node(model, TOOLS))
    graph.add_node("agent", create_node(agent))
    graph.add_edge(START, "superagent")
    graph.add_edge("superagent", "agent")
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
            return
        print()
        stream_graph_updates(graph, user_input, config)


if __name__ == "__main__":
    main()
