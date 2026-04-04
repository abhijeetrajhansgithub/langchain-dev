import os

from dotenv import find_dotenv, load_dotenv
from langchain.agents import create_agent  # type: ignore
from langchain.tools import tool  # type: ignore
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch

load_dotenv(find_dotenv())


_ollama_server = os.getenv("OLLAMA_SERVER")

# -------------------------------------------------------
# -------------------------------------------------------


# -------------------------------------------------------
# -------------------------------------------------------

# gemma3:270m -> 291 MB
# qwen3:4b -> 2.6 GB
# qwen2.5:1.5b -> 986 MB
# qwen3.5:cloud -> cloud

_llm = ChatOllama(
    model="qwen3.5:cloud",
    validate_model_on_init=False,
    temperature=0.2,
    base_url=_ollama_server,
)

_tools = [TavilySearch()]


_agent = create_agent(  # type: ignore
    model=_llm,
    tools=_tools,
)

# -------------------------------------------------------
# -------------------------------------------------------


def main():
    print("Hello from react-search-agent!")

    result = _agent.invoke(  # type: ignore
        {  # type: ignore
            "messages": [HumanMessage(content="What's the weather in Bengaluru?")]
        }
    )

    print("Result: ", result)


if __name__ == "__main__":
    main()
