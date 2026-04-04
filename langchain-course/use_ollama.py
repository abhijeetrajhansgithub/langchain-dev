import os

from dotenv import find_dotenv, load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

load_dotenv(find_dotenv())

_ollama_server = os.getenv("OLLAMA_SERVER")
# gemma3:270m -> 291 MB
# qwen3:4b -> 2.6 GB
# qwen2.5:1.5b -> 986 MB

_llm = ChatOllama(
    model="gemma3:270m",
    validate_model_on_init=False,
    temperature=0.8,
    base_url=_ollama_server,
)

_information = """
Elon Reeve Musk (/ˈiːlɒn/ EE-lon; born June 28, 1971) is a businessman and entrepreneur known for his leadership of Tesla, SpaceX, X, and xAI. Musk has been the wealthiest person in the world since 2025; as of April 2026, Forbes estimates his net worth to be US$823 billion.

Born into a wealthy family in Pretoria, South Africa, Musk emigrated in 1989 to Canada; he has Canadian citizenship since his mother was born there. He received bachelor's degrees in 1997 from the University of Pennsylvania before moving to California to pursue business ventures. In 1995, Musk co-founded the software company Zip2. Following its sale in 1999, he co-founded X.com, an online payment company that later merged to form PayPal, which was acquired by eBay in 2002. Musk also became an American citizen in 2002.
"""

_summary_template = """
Given the information about a person, write a summary for it.
Information: {information}

---
Create the following:
1. A short summary 
2. Two interesting facts about them.
"""


_summary_prompt_template = PromptTemplate(
    input_variables=["information"],  # type: ignore
    template=_summary_template,
)

chain = _summary_prompt_template | _llm  # type: ignore
response = chain.invoke(input={"information": _information})  # type: ignore

print("Response: ", response.content.split("</think>")[-1])  # type: ignore
