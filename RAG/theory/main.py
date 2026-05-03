import os 
from operator import itemgetter
from dotenv import load_dotenv, find_dotenv
from langchain_core.documents import Document

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from langchain_ollama import ChatOllama
from langchain.embeddings import Embeddings

from langchain_pinecone import PineconeVectorStore
import ollama

from typing import Any

load_dotenv(find_dotenv())

EMBEDDING_MODEL = "snowflake-arctic-embed:335m"

print("Initializing components...")

class OllamaEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = ollama.embed(input=texts, model=EMBEDDING_MODEL)
        return response["embeddings"]
    
    def embed_query(self, text: str) -> list[float]:
        response = ollama.embed(input=[text], model=EMBEDDING_MODEL)
        return response["embeddings"][0]
    

ollama_embeddings = OllamaEmbeddings()

llm = ChatOllama(
    model="qwen2.5:1.5b",
    validate_model_on_init=False,
    temperature=0.2,
    base_url=os.getenv("OLLAMA_SERVER"),
)

vector_store = PineconeVectorStore(
    index_name="medium-blogs-embeddings-index",
    embedding=ollama_embeddings
)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

prompt_template = ChatPromptTemplate.from_template(
    """
    Answer the question based only on the following context:

    <context>
    {context}
    </context>

    Question: {question}

    Provide a detailed answer.
    """
)

def format_docs(docs: list[Document]):
    """Format retrieved documents into a single string"""

    return "\n\n".join(doc.page_content for doc in docs)

# ######################################################
# Implementation 1: RAG without LCel
# ######################################################
def retrieval_chain_without_lcel(query: str):
    docs = retriever.invoke(query)

    context = format_docs(docs)

    prompt = prompt_template.format(context=context, question=query)

    return llm.invoke(input=[HumanMessage(content=prompt)])


# ######################################################
# Implementation 2: RAG with LCel
# ######################################################
def create_retrieval_chain_with_lcel():
    """
    Create a retrieval chain using lcel.
    Returns a chain than can be invoked.
    """

    retrieval_chain: Any = (
        RunnablePassthrough.assign(
        context=itemgetter("question") | retriever | RunnableLambda(format_docs)
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )

    return retrieval_chain



if __name__ == "__main__":
    # Query
    query = "What is Pinecone in Machine Learning?"

    # ###########################################################
    # Option 0: Raw invocation without RAG
    # ###########################################################

    print("\n\n" + "="*100)
    print("Option 0: Raw invocation without RAG")
    print("="*100)

    result_raw = llm.invoke(input=[HumanMessage(content=query)])
    print("\nAnswer: \n", result_raw.content)  # type: ignore

    # ###########################################################
    # Option 1: RAG without LCel
    # ###########################################################

    print("\n\n" + "="*100)
    print("Option 1: RAG without LCel")
    print("="*100)

    result_rag = retrieval_chain_without_lcel(query)
    print("\nAnswer: \n", result_rag.content)  # type: ignore

    # ###########################################################
    # Option 2: RAG with LCel
    # ###########################################################

    print("\n\n" + "="*100)
    print("Option 2: RAG with LCel")
    print("="*100)

    retrieval_chain_with_lcel = create_retrieval_chain_with_lcel()
    result_lcel = retrieval_chain_with_lcel.invoke({"question": query})
    print("\nAnswer: \n", result_lcel)

