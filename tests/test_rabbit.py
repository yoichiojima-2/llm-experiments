from langchain.chat_models import init_chat_model
from langchain_community.tools.tavily_search import TavilySearchResults


def test_gpt_4o_mini():
    model = init_chat_model("gpt-4o-mini", model_provider="openai")
    assert model.invoke("hello").content


def test_tavily_search():
    search = TavilySearchResults(max_results=2)
    assert search.invoke("who is the president of the united states")[0]["content"]
