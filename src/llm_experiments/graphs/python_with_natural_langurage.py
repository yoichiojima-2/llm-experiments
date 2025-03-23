import asyncio
from typing import Annotated

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from llm_experiments.agents import PythonAgent
from llm_experiments.llm import create_model


class State(TypedDict):
    messages: Annotated[list[str], add_messages]


def python_executor(state: State):
    model = create_model()
    agent = PythonAgent(model)
    res = agent.agent().invoke({"messages": state["messages"]})
    return {"messages": [agent.get_last_response(res)]}


def graph():
    builder = StateGraph(State)
    builder.add_node("python_executor", python_executor)
    builder.add_edge(START, "python_executor")
    builder.add_edge("python_executor", END)
    return builder.compile(checkpointer=MemorySaver())


async def main(thread_id="python_executor"):
    g = graph()
    config = {"thread_id": thread_id}
    while True:
        user_input = input("user: ")
        res = await g.ainvoke({"messages": [user_input]}, config=config, debug=True)
        print(f"assistant: {res['messages'][-1].content}")


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
