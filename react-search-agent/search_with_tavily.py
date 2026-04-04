from dotenv import load_dotenv, find_dotenv
import os
from langchain.agents import create_agent  # type: ignore
from langchain.tools import tool  # type: ignore
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from tavily import TavilyClient  # type: ignore

_tavily = TavilyClient()

load_dotenv(find_dotenv())

_ollama_server = os.getenv("OLLAMA_SERVER")

# -------------------------------------------------------
# -------------------------------------------------------


@tool
def search(query: str) -> str:
    """
    Tool that searches over the internet
    Args:
        query: The query to search for
    Returns:
        The search result
    """
    print(f"Searching for: {query}")
    
    return _tavily.search(  # type: ignore
        query=query,
    )



# -------------------------------------------------------
# -------------------------------------------------------

# gemma3:270m -> 291 MB
# qwen3:4b -> 2.6 GB
# qwen2.5:1.5b -> 986 MB

_llm = ChatOllama(
    model="qwen2.5:1.5b",
    validate_model_on_init=False,
    temperature=0.2,
    base_url=_ollama_server,
)

_tools = [search]


_agent = create_agent(  # type: ignore
    model=_llm,
    tools=_tools,
)

# -------------------------------------------------------
# -------------------------------------------------------


def main():
    print("Hello from react-search-agent!")

    result = _agent.invoke({  # type: ignore
        "messages": [HumanMessage(content="What's the weather in Tokyo?")]
    })

    print("Result: ", result)


if __name__ == "__main__":
    main()
