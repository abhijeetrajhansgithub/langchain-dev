import os

from dotenv import find_dotenv, load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openrouter import ChatOpenRouter
from pydantic import SecretStr

load_dotenv(find_dotenv())

_model = "qwen/qwen3.6-plus:free"

_key = os.getenv("OPENROUTER_API_KEY")
assert _key is not None

llm = ChatOpenRouter(
    model=_model,
    api_key=SecretStr(_key),
    temperature=0,
    max_tokens=256
)


def main():
    print("Hello from langchain-course!")
    information = """
Elon Reeve Musk (/ˈiːlɒn/ EE-lon; born June 28, 1971) is a businessman and entrepreneur known for his leadership of Tesla, SpaceX, X, and xAI. Musk has been the wealthiest person in the world since 2025; as of April 2026, Forbes estimates his net worth to be US$823 billion.

Born into a wealthy family in Pretoria, South Africa, Musk emigrated in 1989 to Canada; he has Canadian citizenship since his mother was born there. He received bachelor's degrees in 1997 from the University of Pennsylvania before moving to California to pursue business ventures. In 1995, Musk co-founded the software company Zip2. Following its sale in 1999, he co-founded X.com, an online payment company that later merged to form PayPal, which was acquired by eBay in 2002. Musk also became an American citizen in 2002.
"""

    summary_template = """
Given the information about a person, write a summary for it.
Information: {information}

---
Create the following:
1. A short summary 
2. Two interesting facts about them.
"""

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template,
    )

    chain = summary_prompt_template | llm # type: ignore
    response = chain.invoke(input={ # type: ignore
        "information": information
    })

    print("Response: ", response.content)  # type: ignore

    # Without chaining
    # summary = llm.invoke(summary_prompt_template.format(information=information))

    # print the summary
    # print("\n\nSummary:", summary)


if __name__ == "__main__":
    main()
