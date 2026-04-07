from dotenv import load_dotenv, find_dotenv
from typing import List, Literal, Any, Dict  # type: ignore

import requests
import ollama  # type: ignore
import json
from ollama import ChatResponse  # type: ignore

from langsmith import traceable  # type: ignore

load_dotenv(find_dotenv())


MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"


# ------------ tools --------------

def get_product_price(product: str) -> float:
    """
    Look up the price of the product in the catelog
    
    Args:
        product (str): The product to look up.
    Returns:
        float: The price of the product.
    """

    product = product.lower()

    print(f"    -- Looking up the price of {product}... ")


    # Random data for tech
    prices = {
        "laptop": 2000.0,
        "smartphone": 1000.0,
        "tablet": 500.0,
        "headphones": 100.0,
        "keyboard": 50.0,
        "mouse": 30.0,
        "monitor": 200.0,
        "speakers": 150.0,
        "projector": 500.0,
        "camera": 300.0,
        "printer": 200.0,
    }

    return prices.get(product, 0.0)



def apply_discount(price: float, discount_tier: Literal["gold", "silver", "bronze"]) -> float:
    """
    Apply a discount tier to a price and return the final price.

    Args:
        price (float): The original price.
        discount_tier (Literal["gold", "silver", "bronze"]): The discount tier to apply.

    Returns:
        float: The final price after applying the discount.
    """

    print(f"    -- Applying {discount_tier} discount... ")

    print(f"    -- Price before discount: {price}\n    -- Tier: {discount_tier}")

    discount_prcnt = {  # in percent
        "bronze": 5,
        "silver": 10,
        "gold": 20,
    }

    discount = discount_prcnt.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)


# ---------------- generate function json descriptions ----------------
# Difference 1: tool description generated using make_json

from utils import make_json  # type: ignore

_tools_for_llm: List[Any] = [
    get_product_price,
    apply_discount
]

_fn_json_: List[Dict] = make_json(_tools_for_llm)  # type: ignore
print(_fn_json_)  # type: ignore


# ------------------ traced ollama calls --------------------

# Difference 2: ollama_chat_traced is a traced function
@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(messages: List[Dict]):  # type: ignore

    response = requests.post(
        url="http://localhost:11434/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        json={
            "messages": messages,  # type: ignore
            "model": MODEL,
            "tools": _fn_json_,  
        },
    )
    response.raise_for_status()  # raises if 4xx/5xx
    data = response.json()

    return data

# ------------ agent loop --------------


@traceable(name="Langchain Agent Loop")
def run_agent(question: str):  # type: ignore
    print("=" * 55)
    print(f"Question: {question}")
    print("=" * 55)

    _tools = [get_product_price, apply_discount]  # type: ignore
    # Difference 3: _tools_dict is a dictionary
    _tools_dict = {t.__name__: t for t in _tools}  # type: ignore

    print(f"    -- Tools: {', '.join(_tools_dict.keys())}")  # type: ignore


    # Difference 4: _messages is a list of dictionaries instead of SystemMessage and HumanMessage
    _messages: List[Dict[Any, Any]] = [
        { 
            "role": "system",
            "content":("You are a helpful shopping assistant. "
                     "You have access to the following tools: "
                     f"{', '.join(_tools_dict.keys())}"  # type: ignore
                     "STRICT RULES - you must follow these exactly: \n"
                     "1. Never guess or assume any product price. "
                     "2. You must call 'get_product_price' first to get the price of the product. "
                     "3. Call 'apply_discount' AFTER you get the product price from 'get_product_price' to apply the discount. "
                     "4. Do not call 'apply_discount' before 'get_product_price'. "
                     "5. Do not call 'get_product_price' more than once. "
                     "6. Do not call 'apply_discount' more than once. "
                     "7. Pass the exact price into the 'apply_discount' tool -- DO NOT pass a made up price. "
                     "8. Never calculate the final price on your own using math. Always use the 'apply_discount' tool. "
                     "9. If a user doesnot specify a discount tier, ask them to specify a discount tier. DO NOT assume a discount tier. "
                     "10. Important: When in doubt, always check if a tool is available for the task.")
        },
        {
            "role": "user",
            "content": question
        }
    ]


    for iteration in range(1, MAX_ITERATIONS+1):
        print(f" ------- Iteration {iteration} -------")


        # Difference 5: ollama.chat() directly instead of llm_with_tools.invoke()
        response = ollama_chat_traced(messages=_messages)  # type: ignore
        print(f"Response: {response}")

        ai_message = response['choices'][0]['message']  # type: ignore
        print(f"AI Message: {ai_message}")

        try:
            tool_calls = ai_message['tool_calls']  # type: ignore
        except KeyError:
            tool_calls = None

        # if no tool calls, we're done
        if not tool_calls:
            print(f"Final Answer: {ai_message['content']}")  # type: ignore
            return ai_message['content']  # type: ignore

        # tool_calls = ai_message.tool_calls  # type: ignore

        print(f"\n\n\nTool Calls: {tool_calls}")

        num_tool_calls = len(tool_calls)
        print(f"Number of tool calls: {num_tool_calls}")

        # Difference 6: Custom tool call logic is implemented keeping in mid the architecture of Ollama
        for i, tool_call in enumerate(tool_calls):  # type: ignore
            _tool_call = tool_call
            _tool_call_name: str = _tool_call.get("function").get("name")
            _tool_call_args: Any = _tool_call.get("function").get("arguments") 

            print("\n\n[Tool selected] ", _tool_call_name, "\nwith args: ", _tool_call_args)

            _tool_to_use = _tools_dict[_tool_call_name]  # type: ignore

            if _tool_to_use is None:
                raise ValueError(f"Tool {_tool_call_name} not found")
            
            if isinstance(_tool_call_args, str):
                _tool_call_args = json.loads(_tool_call_args)
            
            # Difference 7: Direct function call instead of tool.invoke()
            observation: Any = _tool_to_use(**_tool_call_args)  # type: ignore

            print(f"\n\n[Observation] \n{observation}")

            _messages.append(
                {
                    "role": "assistant",
                    "content": str(ai_message)
                }
            )

            _messages.append(
                {
                    "role": "tool",
                    "name": str(_tool_call_name),
                    "content": str(observation)  # type: ignore
                }
            )

            
    else:
        raise ValueError("Max iterations reached")



if __name__ == "__main__":
    print("Hello from react-agents-under-the-hood!")
    run_agent("What is the price of a laptop after applying the gold discount?")