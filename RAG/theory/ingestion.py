import os 
from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.embeddings import Embeddings
from langchain_pinecone import PineconeVectorStore

from langchain_text_splitters import CharacterTextSplitter
import ollama

load_dotenv(find_dotenv())

CURR_PATH = os.path.dirname(os.path.abspath(__file__))
print("Current Path: ", CURR_PATH)


EMBEDDING_MODEL = "snowflake-arctic-embed:335m"

# ollama.init(model="qwen2.5:1.5b", base_url="http://localhost:11434")

class OllamaEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = ollama.embed(input=texts, model=EMBEDDING_MODEL)
        return response["embeddings"]
    
    def embed_query(self, text: str) -> list[float]:
        response = ollama.embed(input=[text], model=EMBEDDING_MODEL)
        return response["embeddings"][0]


if __name__ == "__main__":
    print("Ingesting...")

    loader = TextLoader(file_path=fr"{CURR_PATH}\mediumblog1.txt", encoding="utf-8")
    docs = loader.load()
    print(docs)

    splitter = CharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )

    docs = splitter.split_documents(docs)
    print("Length of docs: ", len(docs))

    vector_store = PineconeVectorStore(
        index_name="medium-blogs-embeddings-index",
        embedding=OllamaEmbeddings()
    )

    vector_store.add_documents(docs)

    print("Ingestion complete!")

