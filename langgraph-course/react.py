from dotenv import load_dotenv, find_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from typing import List, Any

import os

load_dotenv(find_dotenv())

# LLM_MODEL = "qwen3:1.7b"
LLM_MODEL = "qwen3:4b"

@tool
def triple(num: float) -> float:
    """
    Triple a number

    Args:
        num (float): The number to triple.

    Returns:
        float: The tripled number.
    """
    return float(num) * 3


tools: List[Any] = [TavilySearch(max_results=1), triple]

llm = ChatOllama(
    model=LLM_MODEL,
    validate_model_on_init=False,
    temperature=0.2,
    base_url=os.getenv("OLLAMA_SERVER"),
).bind_tools(tools)     # type: ignore


# response = llm.invoke("What is thrice of 3?")

# print(response)

