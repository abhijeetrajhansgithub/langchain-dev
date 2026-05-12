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

tavily_map = TavilyMap(
    max_depth=5,
    max_breadth=20,
    max_pages=500
)

tavily_extract = TavilyExtract()


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

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

    splitted_docs = text_splitter.split_documents(all_docs)

    logger.info(f"RecursiveCharacterTextSplitter: Successfully chunked {len(splitted_docs)} documents.")





if __name__ == "__main__":
    asyncio.run(main())