import asyncio
from logger import Logger
import os
import ssl 
from typing import List, Dict, Any

import certifi 
from dotenv import load_dotenv, find_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma  # type: ignore
from langchain_pinecone import PineconeVectorStore
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap  # type: ignore

import ollama

load_dotenv(find_dotenv())

logger = Logger(log_file="all.log", color_file=True)

ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

tavily_map = TavilyMap(
    max_depth=5,
    max_breadth=20,
    max_pages=500
)

tavily_extract = TavilyExtract()

EMBEDDING_MODEL = "snowflake-arctic-embed:335m"

class OllamaEmbedding(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = ollama.embed(input=texts, model=EMBEDDING_MODEL)
        return response["embeddings"]
    
    def embed_query(self, text: str) -> List[float]:
        response = ollama.embed(input=[text], model=EMBEDDING_MODEL)
        return response["embeddings"][0]
    

def get_vector_store():
    return PineconeVectorStore(
        index_name="langchain-docs-2026",
        embedding=OllamaEmbedding()
    )


def chunk_urls(urls: List[str], batch_size: int = 10) -> List[List[str]]:
    chunks: List[List[str]] = []

    for i in range(0, len(urls), batch_size):
        chunk = urls[i:i + batch_size]
        chunks.append(chunk)

    return chunks


async def extract_batch(urls: List[str], batch_num: int) -> List[Dict[str, Any]]:
    try:
        docs = await tavily_extract.ainvoke(input={'urls': urls})  # type: ignore
        
        logger.info(f"TavilyExtract: Successfully extracted {len(docs['results'])} URLs from batch {batch_num}.")
        return docs

    except Exception as e:
        logger.error(f"TavilyExtract: Error processing batch {batch_num}: {e}")
        return []
    

async def async_extract(url_batches: List[List[str]]):
    logger.header("DOCUMENT EXTRACTION PIPELINE")

    tasks = [extract_batch(batch, i+1) for i, batch in enumerate(url_batches)]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_pages: List[Any] = []  
    failed_pages = 0 

    for result in results:
        if isinstance(result, Exception):
            logger.error(f"TavilyExtract: Error processing batch {batch_num}: {result}")  # type: ignore
            failed_pages += 1
        else:
            for extracted_page in result['results']:  # type: ignore
                document = Document(  # type: ignore
                    page_content=extracted_page['raw_content'],  # type: ignore
                    metadata={
                        "source": extracted_page['url']
                    }
                )

                all_pages.append(document)  # type: ignore

    logger.info(f"TavilyExtract: Successfully extracted {len(all_pages) - failed_pages} URLs.")
    logger.info(f"TavilyExtract: Failed to extract {failed_pages} URLs.")

    if failed_pages > 0:
        logger.error(f"TavilyExtract: Failed to extract {failed_pages} URLs.")

    return all_pages


async def index_documents_async(
    docs: List[Document],
    batch_size: int = 5,
    max_concurrency: int = 3
):
    logger.header("DOCUMENT INDEXING PIPELINE")

    logger.info(f"Preparing {len(docs)} documents for indexing.")

    batches = [
        docs[i:i + batch_size]
        for i in range(0, len(docs), batch_size)
    ]

    semaphore = asyncio.Semaphore(max_concurrency)

    async def add_batch(batch: List[Document], batch_num: int):
        async with semaphore:
            try:
                vector_store = get_vector_store()

                await vector_store.aadd_documents(batch)

                logger.info(
                    f"Indexed {len(batch)} documents in batch {batch_num}."
                )

                return True

            except Exception as e:
                logger.error(
                    f"Failed to index {len(batch)} documents in batch {batch_num}."
                )

                logger.error(f"Error: {e}")

                return False

    tasks = [
        add_batch(batch, i + 1)
        for i, batch in enumerate(batches)
    ]

    results = await asyncio.gather(
        *tasks,
        return_exceptions=True
    )

    successful_batches = sum(
        1 for result in results if result is True
    )

    failed_batches = sum(
        1 for result in results if result is not True
    )

    logger.info(
        f"Indexed {successful_batches} batches successfully."
    )

    logger.info(
        f"Failed to index {failed_batches} batches."
    )


async def main():
    logger.header("DOCUMENT INGESTION PIPELINE")

    site_map = tavily_map.invoke("https://python.langchain.com/")  # type: ignore
    logger.info(f"TavilyMap: Successfully extracted {len(site_map['results'])} URLs.")

    urls = site_map['results']

    # chunk URLs into batches of 10
    url_chunks = chunk_urls(urls, batch_size=10)

    for chunk in url_chunks:
        logger.info(f"TavilyCrawl: Starting to crawl {len(chunk)} URLs.")

    all_docs = await async_extract(url_chunks)

    logger.info(f"TavilyExtract: Successfully extracted {len(all_docs)} URLs.")


    # Document Chunking Phase
    logger.header("DOCUMENT CHUNKING PIPELINE")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    splitted_docs = text_splitter.split_documents(all_docs)

    logger.info(f"RecursiveCharacterTextSplitter: Successfully chunked {len(splitted_docs)} documents.")


    # Document Indexing Phase
    await index_documents_async(splitted_docs, batch_size=10)


if __name__ == "__main__":
    asyncio.run(main())