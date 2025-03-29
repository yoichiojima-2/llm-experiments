import sys

from langgraph.prebuilt import create_react_agent

from llm_experiments import tools
from llm_experiments.llm import create_model


def main():
    try:
        model = create_model("gpt-o3-mini")
        agent = create_react_agent(model, tools.shell())
        res = agent.invoke({"messaages": "execute git diff and create commit message"})

        for i in res["messages"]:
            print(i.content)

    except Exception as e:
        print(f"failed generating commit messages: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
