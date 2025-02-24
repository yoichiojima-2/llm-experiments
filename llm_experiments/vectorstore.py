import asyncio
import os

from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader

from llm_experiments.embedding import instantiate_embedding
from llm_experiments.utils import get_app_root


class VectorStore:
    def __init__(
        self,
        collection_name="default",
        persist_directory=str(get_app_root() / ".vectorstore"),
        embedding="nomic-embed-text",
    ):
        print(f"Initializing VectorStore with collection: {collection_name}, directory: {persist_directory}")
        self.vs = Chroma(
            collection_name=collection_name,
            embedding_function=instantiate_embedding(embedding),
            persist_directory=persist_directory,
        )

    async def aadd(self, docs):
        print(f"Adding {len(docs)} documents...")
        await self.vs.aadd_documents(docs)
        result = self.get_all()
        print(f"After adding, collection has {len(result['ids'])} documents")

    async def add_webpages(self, urls):
        print(f"Fetching webpages: {urls}")
        loader = WebBaseLoader(web_paths=urls)
        async for doc in loader.alazy_load():
            await self.aadd([doc])

    def get_all(self):
        result = self.vs.get()
        print(f"Getting all documents: found {len(result['ids']) if result['ids'] else 0} documents")
        return result


async def main():
    vs = VectorStore()
    print("\nAdding webpage...")
    await vs.add_webpages(["https://python.langchain.com/docs/concepts/tool_calling/"])

    print("\nVerifying contents:")
    contents = vs.get_all()
    print(f"Number of documents: {len(contents['ids'])}")

    print(f"Current working directory: {os.getcwd()}")

    print("\nFiles in .vectorstore directory:")
    if os.path.exists(".vectorstore"):
        print(os.listdir(".vectorstore"))
    else:
        print(".vectorstore directory not found")


if __name__ == "__main__":
    asyncio.run(main())
