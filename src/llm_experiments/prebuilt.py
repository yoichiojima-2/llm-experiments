from llm_experiments import tools as t
from llm_experiments.agent import Agent
from llm_experiments.multiagents.swe import SWETeam


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
        tools=[
            *t.Tavily().get_tools(),
            *t.DuckDuckGo().get_tools(),
            *t.Serper().get_tools(),
            *t.Wikipedia().get_tools(),
        ],
        memory=memory,
        config=config,
    )


async def shell(model, memory, config):
    return Agent(
        model=model,
        tools=t.Shell().get_tools(ask_human_input=True),
        memory=memory,
        config=config,
    )


async def shell_w_search(model, memory, config):
    return Agent(
        model=model,
        tools=[
            *t.Shell().get_tools(ask_human_input=True),
            *t.Tavily().get_tools(),
            *t.DuckDuckGo().get_tools(),
            *t.Serper().get_tools(),
        ],
        memory=memory,
        config=config,
    )


async def slack(model, memory, config):
    return Agent(
        model=model,
        tools=t.Slack().get_tools(),
        memory=memory,
        config=config,
    )


async def python_repl(model, memory, config):
    return Agent(
        model=model,
        tools=t.Python_().get_tools(),
        memory=memory,
        config=config,
    )


async def sql(model, memory, config):
    return Agent(
        model=model,
        tools=[*t.SQL().get_tools(model, "sql"), t.Shell().get_tools(), t.DuckDuckGo().get_tools()],
        memory=memory,
        config=config,
    )


async def browser(model, memory, config, browser):
    return Agent(
        model=model,
        tools=t.Browser(browser).get_tools(),
        memory=memory,
        config=config,
    )


async def browser_w_search(model, memory, config, browser):
    return Agent(
        model=model,
        tools=[
            *t.Browser(browser).get_tools(),
            *t.DuckDuckGo().get_tools(),
            *t.Serper().get_tools(),
            *t.Wikipedia().get_tools(),
            *t.Tavily().get_tools(),
        ],
        memory=memory,
        config=config,
    )
