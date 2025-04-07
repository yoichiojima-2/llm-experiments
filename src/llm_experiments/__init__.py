from langgraph.checkpoint.memory import MemorySaver

from llm_experiments import tools
from llm_experiments.agent import Agent
from llm_experiments.clients import Config, create_executor
from llm_experiments.llm import create_model

model = create_model()
toolkit = tools.slack_tools()
config = Config(
    model=model,
    memory=MemorySaver(),
    verbose=True,
    configurable={"configurable": {"thread_id": "default"}},
)
executor = create_executor(config.model, toolkit, config.verbose)
agent = Agent(
    executor=executor,
    model=config.model,
    memory=config.memory,
    tools=toolkit,
    verbose=config.verbose,
    config=config.configurable,
)
