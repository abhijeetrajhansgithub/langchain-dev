import re 
import inspect 
from typing import Any, Dict, List, Literal
import requests

from langsmith import traceable  # type: ignore


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

    if isinstance(price, str):
        price = float(price)

    discount_prcnt = {  # in percent
        "bronze": 5,
        "silver": 10,
        "gold": 20,
    }

    discount = discount_prcnt.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)


# -------------- tools dict ----------------
tools: dict[str, Any] = {
    "get_product_price": get_product_price,
    "apply_discount": apply_discount
}


# ------------ get tool descriptions --------------

def get_tool_description(tool_dict: dict[str, Any]) -> str:
    descriptions: list[str] = []

    for name, tool in tool_dict.items():
        original_function = getattr(tool, "__wrapped__", tool)
        signature = inspect.signature(original_function)
        docstring = inspect.getdoc(original_function)

        descriptions.append(f"{name}: {signature} - {docstring}")

    return "\n\n---\n".join(descriptions)


tool_descriptions = get_tool_description(tools)
tool_names = ", ".join(tools.keys())
print(tool_descriptions)


# --------------------- react prompt ---------------------------

react_prompt = """
STRICT RULES — you must follow these exactly:
1. NEVER guess or assume any product price. You MUST call get_product_price first to get the real price.
2. Only call apply_discount AFTER you have received a price from get_product_price. Pass the exact price returned by get_product_price — do NOT modify it.
3. NEVER calculate discounts yourself using math. Always use the apply_discount tool.
4. If the user does not specify a discount tier, ask them which tier to use — do NOT assume one.

Answer the following questions as best you can. You have access to the following tools:

{tool_descriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action, as comma separated values
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {question}
Thought:
"""



# ------------------ traced ollama calls --------------------

# Difference 2: ollama_chat_traced is a traced function
@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(messages: List[Dict], options) -> Dict:  # type: ignore

    """
    Call the Ollama API to get responses to a chat prompt

    Args:
        messages (List[Dict]): The messages to send to the Ollama chat
        options (dict): The options to pass to the Ollama API

    Returns:
        Dict: The response from the Ollama API
    """
    response = requests.post(  # type: ignore
        url="http://localhost:11434/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        json={
            "messages": messages,  # type: ignore
            "model": MODEL,
            "options": options,  
            "stop": ["Observation:", "\nObservation:", "\nObservation"]
        },
    )
    response.raise_for_status()  # types: ignore
    data = response.json()  # type: ignore

    return data


# -------------- agent loop --------------

@traceable(name="Ollama ReAct Agent Loop")
def run_agent(question: str):

    print("=" * 55)
    print(f"Question: {question}")
    print("=" * 55)

    prompt: str = react_prompt.format(
        question=question,
        tool_descriptions=tool_descriptions,
        tool_names=tool_names
    )
    scratchpad = ""

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"--- Iteration {iteration} ---")

        full_prompt = prompt + scratchpad
        
        response: Dict[Any, Any] = ollama_chat_traced(  # type: ignore
            messages=[
                {"role": "user", "content": full_prompt},
            ],
            options={
                "temperature": 0,
            }
        )

        output = response['choices'][0]['message']['content']  # type: ignore
        print(f"LLM Output: {output}")

        print("[PARSING] Looking for final answer...")
        final_answer_match = re.search(r"\nFinal Answer:\s*(.+)", output)
        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()

            print("=" * 55)
            print("Final Answer Found!")
            print(f"Final Answer: {final_answer}")
            print("=" * 55)

            return final_answer
        

        print("[PARSING] Looking for Action and Action Input in LLM Output")
        action_match = re.search(r"\nAction:\s*(.+)", output)
        action_input_match = re.search(r"\nAction Input:\s*(.+)", output)

        if not action_match or not action_input_match:
            print("[PARSER] No Action or Action Input found in LLM Output. Exiting.")
            
            scratchpad += f"{output}\nObservation: Error: No Action or Action Input found in LLM Output. Exiting.\nThought:"
            continue

        tool_name = action_match.group(1).strip()
        tool_input_raw = action_input_match.group(1).strip()

        print(f"Tool Name: {tool_name}")
        print(f"Tool Input: {tool_input_raw}")

        raw_args: list[str] = [x.strip() for x in tool_input_raw.split(",")]
        args = [x.split("=", 1)[-1].strip().strip("'\"") for x in raw_args]

        print(f"[Tool Execution] Executing {tool_name} with {args}")

        if tool_name not in tools:
            observation = f"Error: '{tool_name}' is not a valid tool. Available tools: {', '.join(tools.keys())}"

        else:
            observation = str(tools[tool_name](*args))  # type: ignore
        
        
        print(f"[Tool Execution] Observation: {observation}")

        scratchpad += f"{output}\nObservation: {observation}\nThought:"
    
    else:
        print("Max iterations reached. No final answer found.")
        return None




if __name__ == "__main__":
    print("Hello from react-prompt-function-calling!")
    run_agent("What is the price of a laptop after applying the gold discount?")