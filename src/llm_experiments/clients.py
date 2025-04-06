from argparse import ArgumentParser

from llm_experiments import tools
from llm_experiments.agent import Agent
from llm_experiments.llm import create_model


def parse_args():
    parser = ArgumentParser()
    opt = parser.add_argument
    opt("--agent", "-a", choices=["search"])
    opt("--model", "-m", type=str, default="4o-mini")
    opt("--verbose", "-v", action="store_true", default=False)
    return parser.parse_args()


def search(model, verbose, config):
    agent = Agent(create_model(model), [tools.tavily(), tools.duckduckgo(), tools.serper()], verbose=verbose, config=config)
    agent.interactive_chat()


def main():
    config = {"configurable": {"thread_id": "search"}}
    args = parse_args()
    match args.agent:
        case "search":
            search(args.model, args.verbose, config)
        case _:
            raise ValueError(f"Unknown agent: {args.agent}")


main()
