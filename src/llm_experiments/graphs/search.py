import textwrap
from typing import Literal

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.types import Command

from llm_experiments import prompts, tools
from llm_experiments.graphs.utils import create_node, parse_args, stream_graph_updates
from llm_experiments.llm import create_model

TOOLS = [tools.duckduckgo(), tools.serper(), tools.tavily()]


def superagent(state: MessagesState) -> Command[Literal["agent", "__end__"]]:
    model = create_model()
    system_prompt = textwrap.dedent(
        f"""
        Answer the following questions as best you can
        You have access to the following tools:
        {TOOLS}
        """
    )

    msgs = [SystemMessage(content=system_prompt), *state["messages"]]
    res = model.bind_tools(TOOLS).invoke(msgs)

    if len(res.tool_calls) > 0:
        tool_call_id = res.tool_calls[-1]["id"]
        tool_msg = {
            "role": "tool",
            "content": "Successfully transferred",
            "tool_call_id": tool_call_id,
        }
        return Command(goto="agent", update={"messages": [res, tool_msg]})


def create_agent(model, verbose=False) -> None:
    prompt = prompts.multipurpose()
    agent = create_react_agent(model, TOOLS, prompt=prompt)
    return AgentExecutor(agent=agent, tools=TOOLS, handle_parsing_errors=True, verbose=verbose)


def create_graph(model, verbose):
    graph = StateGraph(MessagesState)
    agent = create_agent(model, verbose=verbose)
    graph.add_node("super_agent", superagent)
    graph.add_node("agent", create_node(agent))
    graph.add_edge(START, "super_agent")
    graph.add_edge("super_agent", "agent")
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
        print()
        stream_graph_updates(graph, user_input, config)


if __name__ == "__main__":
    main()
