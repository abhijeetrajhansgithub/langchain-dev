from dotenv import load_dotenv, find_dotenv
import os

from langchain.agents import create_agent  # type: ignore
from langchain.tools import tool  # type: ignore
from langchain.messages import HumanMessage

from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch


from typing import List
from pydantic import BaseModel, Field

load_dotenv(find_dotenv())

'''
Structured Output:

Add the parameter: response_format
to create_agent runnable
'''

class Source(BaseModel):
    """Schema for a source used by an agent"""

    url: str = Field(description="The url of the source")


class AgentResponse(BaseModel):
    """Schema for the agent response"""

    answer: str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(default_factory=list, description="The sources used by the agent")  # type: ignore



_llm = ChatOllama(
    base_url=os.getenv("OLLAMA_SERVER"),
    model="qwen3.5:cloud",
    validate_model_on_init=False,
    temperature=0.2,
)

_tools = [TavilySearch()]
_agent = create_agent(  # type: ignore
    model=_llm,
    tools=_tools,
    response_format=AgentResponse
)


def main():
    print("Hello from react-search-agent!")

    result = _agent.invoke(  # type: ignore
        {  # type: ignore
            "messages": [
                HumanMessage(content="""
                    Search for jobs in LangChain and LangGraph.

                    Return ONLY JSON in this format:
                    {
                    "answer": "...",
                    "sources": [{"url": "..."}]
                    }

                    You MUST include sources from the search tool.
                """)
            ]
        }
    )

    print("Result: ", result)


if __name__ == "__main__":
    main()