from dotenv import load_dotenv, find_dotenv
import os 

from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState, StateGraph, END

from nodes import run_agent_reasoning, tool_node

AGENT_REASON = "agent_reason"
ACT = "act"
LAST = -1

def should_continue(state: MessagesState):
    last_message = state["messages"][-1]

    if not last_message.tool_calls:  # type: ignore
        return END

    return ACT

load_dotenv(find_dotenv())

flow = StateGraph(MessagesState)

flow.add_node(AGENT_REASON, run_agent_reasoning)  # type: ignore
flow.set_entry_point(AGENT_REASON)
flow.add_node(ACT, tool_node)   # type: ignore

flow.add_conditional_edges(AGENT_REASON, should_continue, {
    END: END,
    ACT: ACT
})

flow.add_edge(ACT, AGENT_REASON)

app = flow.compile()    # type: ignore
app.get_graph().draw_mermaid_png(output_file_path="graph.png")

def main():
    print("Hello from langgraph-course!")
    res = app.invoke({  # type: ignore
        "messages": [HumanMessage(content="What's the weather in Bengaluru? List it and then triple it.")]
    })

    print("Result: ", res)
    print(res["messages"][LAST].content)


if __name__ == "__main__":
    main()
