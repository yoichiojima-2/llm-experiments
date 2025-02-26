import asyncio
import os

from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader

from llm_experiments.models import instantiate_embedding
from llm_experiments.utils import get_app_root


URLS = [
    "https://python.langchain.com/docs/concepts/tool_calling/"
]

class VectorStore:
    def __init__(
        self,
        collection_name="default",
        persist_directory=str(get_app_root() / ".vectorstore"),
        embedding="nomic-embed-text",
    ):
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

    def get_all(self):
        result = self.vs.get()
        return result


async def main():
    vs = VectorStore()
    await vs.add_webpages(URLS)
    print(vs.get_all())

if __name__ == "__main__":
    asyncio.run(main())
