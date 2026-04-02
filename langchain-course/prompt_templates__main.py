from dotenv import find_dotenv, load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_openrouter import ChatOpenRouter

from langsmith import SecretStr

load_dotenv(find_dotenv())

_model = "qwen/qwen3.6-plus:free"

llm = ChatOpenRouter(
    model=_model,
    api_key=SecretStr(os.getenv("OPENROUTER_API_KEY")),
    temperature=0,
)


def main():
    print("Hello from langchain-course!")


if __name__ == "__main__":
    main()
