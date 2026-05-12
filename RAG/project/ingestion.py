import asyncio
from logger import Logger
import os
import ssl 
from typing import List, Dict, Any

import certifi 
from dotenv import load_dotenv, find_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_pinecone import PineconeVectorStore
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap

import ollama

load_dotenv(find_dotenv())

logger = Logger(log_file="all.log", color_file=True)

ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


EMBEDDING_MODEL = "snowflake-arctic-embed:335m"


class OllamaEmbedding(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = ollama.embed(input=texts, model=EMBEDDING_MODEL)
        return response["embeddings"]
    

    def embed_query(self, text: str) -> List[float]:
        response = ollama.embed(input=[text], model=EMBEDDING_MODEL)
        return response["embeddings"][0]
    

embeddings = OllamaEmbedding()

vector_store = PineconeVectorStore(
    index_name="langchain-docs-2026",
    embedding=embeddings
)

tavily_extract = TavilyExtract()
tavily_map = TavilyMap(
    max_depth=5,
    max_breadth=20,
    max_pages=1000
)

tavily_crawl = TavilyCrawl()


async def main():
    logger.header("Ingestion")

    logger.info(f"TavilyCrawl: Starting to crawl documentation from https://python.langchain.com/")

    res = tavily_crawl.invoke({  # type: ignore
        "url": "https://python.langchain.com/",
        "msx_depth": 5,
        "extract_depth": "advanced",
        # "instructions": "content on ai agents"
    })

    all_docs = [
        Document(
            page_content=result['raw_content'], 
            metadata={
                "source": result['url'] 
            }) 
        
        for result in res['results']
    ]

    logger.info(f"TavilyCrawl: Successfully crawled {len(all_docs)} URLs.")

    




if __name__ == "__main__":
    asyncio.run(main())