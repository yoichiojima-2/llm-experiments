import pytest
from llm_experiments.vectorstore import VectorStore


@pytest.mark.asyncio
async def test_vectorstore():
    vs = VectorStore(collection_name="test")
    await vs.add_webpages(["https://python.langchain.com/docs/concepts/tool_calling/"])
    assert vs.vs.get() is not None
    assert vs.vs.as_retriever().invoke("how can I test tool")[0].page_content is not None