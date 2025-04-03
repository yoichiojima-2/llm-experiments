from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_react_agent

from llm_experiments import prompts, tools
from llm_experiments.agents.utils import parse_args
from llm_experiments.llm import create_model


def agent(model, verbose=False) -> None:
    prompt = prompts.multipurpose()
    tool_list = [tools.wikipedia()]
    agent = create_react_agent(model, tool_list, prompt=prompt)
    return AgentExecutor(agent=agent, tools=tool_list, handle_parsing_errors=True, verbose=verbose)


def main():
    load_dotenv()
    args = parse_args()
    model = create_model(args.model)
    for i in agent(model, verbose=args.verbose).stream({"input": args.query}):
        print(i["messages"][-1].content, end="\n\n")


if __name__ == "__main__":
    main()
