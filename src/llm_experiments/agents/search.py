from argparse import ArgumentParser

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_react_agent

from llm_experiments import prompts, tools
from llm_experiments.llm import create_model


def agent(model, verbose=False) -> None:
    prompt = prompts.multipurpose()
    tool_list = [
        tools.duckduckgo(),
        tools.tavily(),
        tools.serper(),
    ]
    agent = create_react_agent(model, tool_list, prompt=prompt)
    return AgentExecutor(agent=agent, tools=tool_list, handle_parsing_errors=True, verbose=verbose)


def parse_args():
    parser = ArgumentParser()
    opt = parser.add_argument
    opt("--query", "-q", type=str, required=True)
    opt("--model", "-m", type=str, default="4o-mini")
    opt("--verbose", "-v", action="store_true", default=False)
    return parser.parse_args()


def main():
    load_dotenv()
    args = parse_args()
    model = create_model(args.model)
    for i in agent(model, verbose=args.verbose).stream({"input": args.query}):
        print(i["messages"][-1].content, end="\n\n")


if __name__ == "__main__":
    load_dotenv()
    main()
