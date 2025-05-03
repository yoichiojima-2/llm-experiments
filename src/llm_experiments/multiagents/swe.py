from pathlib import Path

from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from llm_experiments import tools as t
from llm_experiments.agent import Agent, AgentBase


class SWETeam(AgentBase):
    def __init__(self, model, memory, config, workdir="output/swe"):
        self.model = model
        self.memory = memory
        self.config = config

        self.workdir = Path(workdir)
        if not self.workdir.exists():
            self.workdir.mkdir(parents=True, exist_ok=True)

        self.tools = [
            self.designer_node,
            self.programmer_node,
            self.reviewer_node,
            self.tester_node,
            self.researcher_node,
            *t.Shell().get_tools(),
        ]
        self.graph = self.compile_graph()

    def compile_graph(self):
        g = StateGraph(MessagesState)
        g.add_node("lead", self.lead_node)
        g.add_node("members", ToolNode([*self.tools]))
        g.add_edge(START, "lead")
        g.add_edge("members", "lead")
        g.add_conditional_edges("lead", self.branch_node, ["members", END])
        return g.compile(checkpointer=self.memory)

    @property
    def lead_node(self):
        def lead(state: MessagesState):
            payload = [
                SystemMessage(
                    content=(
                        "You are a software engineering team lead. "
                        "You will be responsible for leading the team and ensuring that the system meets the requirements. "
                        "lead members specialized in research, design, programming, reviewing, and testing. "
                    )
                ),
                *state["messages"],
            ]
            res = self.model.bind_tools(self.tools).invoke(payload)
            return {"messages": [res]}

        return lead

    @property
    def researcher_node(self):
        @tool
        def researcher(state: MessagesState):
            """
            research anything that is needed to complete the task.
            """
            agent = Agent(
                self.model,
                [t.Tavily(), t.DuckDuckGo(), t.Serper()],
                self.memory,
                self.config,
            )
            res = agent.invoke({"messages": state["messages"]})
            return {"messages": res["messages"]}

        return researcher

    @property
    def designer_node(self):
        @tool
        def designer(state: MessagesState):
            """
            design the system and ensuring that it meets the requirements.
            """
            agent = Agent(
                self.model,
                [t.Tavily(), t.DuckDuckGo(), t.Serper()],
                self.memory,
                self.config,
            )
            res = agent.invoke({"messages": state["messages"]})
            return {"messages": res["messages"]}

        return designer

    @property
    def programmer_node(self):
        @tool
        def programmer(state: MessagesState):
            """
            write the code and ensuring that it meets the requirements.
            """
            toolkit = [t.DuckDuckGo(), t.Shell()]
            agent = Agent(self.model, toolkit, self.memory, self.config)
            res = agent.invoke({"messages": state["messages"]})
            return {"messages": res["messages"]}

        return programmer

    @property
    def reviewer_node(self):
        @tool
        def reviewer(state: MessagesState):
            """
            review the code and ensuring that it meets the requirements.
            """
            toolkit = [t.DuckDuckGo(), t.Shell()]
            agent = Agent(self.model, toolkit, self.memory, self.config)
            res = agent.invoke({"messages": state["messages"]})
            return {"messages": res["messages"]}

        return reviewer

    @property
    def tester_node(self):
        @tool
        def tester(state: MessagesState):
            """
            test the code and ensuring that it meets the requirements.
            """
            toolkit = [
                t.Tavily(),
                t.DuckDuckGo(),
                t.Serper(),
                t.Shell(),
                t.Python_(),
            ]
            agent = Agent(self.model, toolkit, self.memory, self.config)
            res = agent.invoke({"messages": state["messages"]})
            return {"messages": res["messages"]}

        return tester

    @property
    def branch_node(self):
        def should_continue(state: MessagesState):
            if state["messages"][-1].tool_calls:
                return "members"
            return END

        return should_continue
