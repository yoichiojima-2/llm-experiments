import nest_asyncio

from llm_experiments import tools as t
from llm_experiments.agent import Agent
from llm_experiments.multiagents.swe import SWETeam

nest_asyncio.apply()


async def swe(model, memory, config, workdir="output/swe"):
    return SWETeam(
        model=model,
        memory=memory,
        config=config,
        workdir=workdir,
    )


async def search(model, memory, config):
    return Agent(
        model=model,
        tools=[t.tavily(), t.duckduckgo(), t.serper(), t.wikipedia()],
        memory=memory,
        config=config,
    )


async def shell_w_search(model, memory, config):
    return Agent(
        model=model,
        tools=[t.shell(ask_human_input=True), t.tavily(), t.duckduckgo(), t.serper()],
        memory=memory,
        config=config,
    )


async def shell(model, memory, config):
    return Agent(
        model=model,
        tools=[t.shell(ask_human_input=True)],
        memory=memory,
        config=config,
    )


async def slack(model, memory, config):
    return Agent(
        model=model,
        tools=t.slack_tools(),
        memory=memory,
        config=config,
    )


async def python_repl(model, memory, config):
    return Agent(
        model=model,
        tools=[t.python_repl()],
        memory=memory,
        config=config,
    )


async def sql(model, memory, config):
    return Agent(
        model=model,
        tools=[*t.sql_tools(model, "sql"), t.shell(), t.duckduckgo()],
        memory=memory,
        config=config,
    )


async def browser(model, memory, config, browser):
    toolkit = await t.browser_tools(browser)
    return Agent(
        model=model,
        tools=toolkit,
        memory=memory,
        config=config,
    )


async def browser_w_search(model, memory, config, browser):
    browser_tools = await t.browser_tools(browser)
    return Agent(
        model=model,
        tools=[*browser_tools, t.duckduckgo(), t.serper(), t.wikipedia(), t.tavily()],
        memory=memory,
        config=config,
    )
