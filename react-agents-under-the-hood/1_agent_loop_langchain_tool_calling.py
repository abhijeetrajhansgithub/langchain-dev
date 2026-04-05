from dotenv import load_dotenv, find_dotenv
from typing import List, Literal, Any  # type: ignore

from langchain.chat_models import init_chat_model
from langchain.tools import tool  # type: ignore
from langchain.messages import HumanMessage, SystemMessage, ToolMessage  # type: ignore

from langsmith import traceable  # type: ignore

load_dotenv(find_dotenv())


MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"


# ------------ tools --------------

@tool
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


@tool
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


# ------------ agent loop --------------


@traceable(name="Langchain Agent Loop")
def run_agent(question: str):  # type: ignore
    print("=" * 55)
    print(f"Question: {question}")
    print("=" * 55)

    _tools = [get_product_price, apply_discount]
    _tools_dict = {t.name: t for t in _tools}

    llm = init_chat_model(f"ollama:{MODEL}", temperature=0)
    llm_with_tools = llm.bind_tools(tools=_tools)  # type: ignore


    _messages: List[Any] = [
        SystemMessage(
            content=("You are a helpful shopping assistant. "
                     "You have access to the following tools: "
                     f"{', '.join(_tools_dict.keys())}"
                     "STRICT RULES - you must follow these exactly: \n"
                     "1. Never guess or assume any product price. "
                     "2. You must call 'get_product_price' first to get the price of the product. "
                     "3. Call 'apply_discount' AFTER you get the product price from 'get_product_price' to apply the discount. "
                     "4. Do not call 'apply_discount' before 'get_product_price'. "
                     "5. Do not call 'get_product_price' more than once. "
                     "6. Do not call 'apply_discount' more than once. "
                     "7. Pass the exact price into the 'apply_discount' tool -- DO NOT pass a made up price. "
                     "8. Never calculate the final price on your own using math. Always use the 'apply_discount' tool. "
                     "9. If a user doesnot specify a discount tier, ask them to specify a discount tier. DO NOT assume a discount tier. ")
        ),
        HumanMessage(content=question)
    ]


    for iteration in range(1, MAX_ITERATIONS+1):
        print(f" ------- Iteration {iteration} -------")

        ai_message = llm_with_tools.invoke(_messages)

        tool_calls = ai_message.tool_calls

        # if no tool calls, we're done
        if not tool_calls:
            print(f"Final Answer: {ai_message.content}")  # type: ignore
            return ai_message.content  # type: ignore
        
        _tool_call = tool_calls[0]
        _tool_name = _tool_call.get("name")
        _tool_args = _tool_call.get("args", {})
        _tool_call_id = _tool_call.get("id")

        print(f"[Tool Selected] {_tool_name} with args: {_tool_args}")

        _tool_to_use = _tools_dict.get(_tool_name)

        if _tool_to_use is None:
            raise ValueError(f"Tool {_tool_name} not found")
        
        observation = _tool_to_use.invoke(input=_tool_args)  # type: ignore

        print(f"   [Tool result] {observation}")

        _messages.append(ai_message)
        _messages.append(
            ToolMessage(
                content=str(observation),
                tool_call_id=_tool_call_id
            )
        )

    else:
        raise ValueError("Max iterations reached")



if __name__ == "__main__":
    print("Hello from react-agents-under-the-hood!")
    run_agent("What is the price of a laptop after applying the gold discount?")