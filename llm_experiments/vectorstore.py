import bs4
import asyncio
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from llm_experiments.embedding import instantiate_embedding
from langchain_core.tools import tool


class VectorStore:
    def __init__(self, collection_name="default", persist_directory=".vectorstore", embedding="nomic-embed-text"):
        self.vs = Chroma(
            collection_name=collection_name,
            embedding_function=instantiate_embedding(embedding),
            persist_directory=persist_directory,
        )

    async def aadd(self, docs):
        await self.vs.aadd_documents(docs)

    async def add_webpages(self, urls):
        loader = WebBaseLoader(web_paths=urls)
        async for doc in loader.alazy_load():
            await self.aadd([doc])
    
    def as_retriever(self):
        return self.vs.as_retriever()

    @tool
    def tool(self, query: str) -> str:
        """
        Retrieve documents from the VectorStore based on the query.
        """
        return self.vs.as_retriever().invoke(query)

async def main():
    vs = VectorStore()
    await vs.add_webpages(["https://python.langchain.com/docs/concepts/tool_calling/"])


if __name__ == "__main__":
    asyncio.run(main())