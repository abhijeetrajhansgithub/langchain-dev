import os 
from dotenv import load_dotenv, find_dotenv

from typing import List, Dict, Any, Tuple

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings import Embeddings
from langchain.messages import ToolMessage

import ollama

load_dotenv(find_dotenv())

MODEL = "qwen3:1.7b"
EMBEDDING_MODEL = "snowflake-arctic-embed:335m"


class OllamaEmbeddings(Embeddings):
    def embed_query(self, text: str) -> list[float]:
        response = ollama.embed(input=[text], model=EMBEDDING_MODEL)
        return response["embeddings"][0]
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = ollama.embed(input=texts, model=EMBEDDING_MODEL)
        return response["embeddings"]
    

embeddings = OllamaEmbeddings()

vector_store = PineconeVectorStore(
    index_name="langchain-docs-2026",
    embedding=embeddings
)

_model = init_chat_model("qwen3:1.7b", base_url="http://localhost:11434", model_provider="ollama")


# --------------------------------------------------------------------
# --------------------------------------------------------------------

@tool(response_format="content_and_artifact")
def retrieve_context(query: str) -> Tuple[str, List[Any]]:
    """
    Retrieve relevant context from the vector store for a given query.

    Args:
        query (str): The query to retrieve context for.

    Returns:
        Tuple[str, List[Any]]: A tuple containing the retrieved context and the retrieved documents.
    """

    retrieved_docs = vector_store.as_retriever().invoke(query, k=5)

    serialized: str = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')}\n\nContent: {doc.page_content}"  # type: ignore
        for doc in retrieved_docs
    )

    return serialized, retrieved_docs  


def run_llm(query: str) -> Dict[str, Any]:
    """
    Run the RAG pipeline to answer a query using the retrieved context

    Args:
        query (str): The query to answer

    Returns:
        Dict[str, Any]: The response from the LLM
    """

    system_prompt = (
    "You are a helpful AI assistant that answers questions about LangChain documentation. "
    "You have access to a tool that retrieves relevant documentation. "
    "Use the tool to find relevant information before answering questions. "
    "Always cite the sources you use in your answers. "
    "If you cannot find the answer in the retrieved documentation, say so."
    )   

    agent = create_agent(
        model=_model,
        tools=[retrieve_context],
        system_prompt=system_prompt,
    )

    # Build messages list
    messages = [{
        'role': 'user',
        'content': query
    }]

    response = agent.invoke({  # type: ignore
        'messages': messages
    })

    answer = response['messages'][-1].content

    context_docs: List[Any] = []

    for message in response['messages']:
        if isinstance(message, ToolMessage) and hasattr(message, 'artifact'):
            context_docs.extend(message.artifact)

    
    return {
        "answer": answer,
        "context": context_docs
    }


if __name__ == "__main__":

    result = run_llm("What are deep agents?")

    print(result)