from argparse import ArgumentParser

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_react_agent

from llm_experiments import prompts, tools
from llm_experiments.llm import create_model


def parse_args():
    parser = ArgumentParser()
    opt = parser.add_argument
    opt("--query", "-q", type=str, required=True)
    opt("--model", "-m", type=str, default="4o-mini")
    opt("--verbose", "-v", action="store_true", default=False)
    return parser.parse_args()


def main(query: str, model_opt: str, verbose=False) -> None:
    model = create_model(model_opt)
    prompt = prompts.multipurpose()

    tool_list = [tools.duckduckgo(), tools.wikipedia(), tools.tavily()]

    agent = create_react_agent(model, tool_list, prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tool_list, handle_parsing_errors=True, verbose=verbose)

    res = agent_executor.stream({"input": query})
    for i in res:
        print(i["messages"][-1].content, end="\n\n")


if __name__ == "__main__":
    load_dotenv()
    args = parse_args()
    main(args.query, args.model, verbose=args.verbose)
