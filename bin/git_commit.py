import sys
import textwrap

from langgraph.prebuilt import create_react_agent

from llm_experiments import tools
from llm_experiments.llm import create_model




def main():
    try:
        prompt = textwrap.dedent(
            """
            your task is create git commit with concise and readable commit messages with given shell tool.
            inspect git diff and git add by lines, then git commit repeatedly to avoid making too large or multi-purpose commits.
            """
        )
        model = create_model("gpt-o3-mini")
        agent = create_react_agent(model, tools.shell())
        res = agent.invoke({"messages": prompt})

        for i in res["messages"]:
            print(i.content)

    except Exception as e:
        print(f"failed generating commit messages: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
