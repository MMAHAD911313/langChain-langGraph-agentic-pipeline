from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_core.tools import tool

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition


# =========================
# 🧠 1. Define LLM
# =========================
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)


# =========================
# 🛠️ 2. Define Tools
# =========================
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


tools = [multiply]

llm_with_tools = llm.bind_tools(tools)


# =========================
# 📦 3. Define Graph State
# =========================
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], "conversation"]


# =========================
# 🤖 4. Agent Node (LLM)
# =========================
def agent_node(state: AgentState):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": messages + [response]}


# =========================
# 🔧 5. Tool Node
# =========================
tool_node = ToolNode(tools=[multiply])


# =========================
# 🔁 6. Routing Logic
# =========================
def should_continue(state: AgentState):
    last_message = state["messages"][-1]

    # If LLM calls tool → go tool node
    if getattr(last_message, "tool_calls", None):
        return "tools"

    return END


# =========================
# 🧩 7. Build Graph
# =========================
workflow = StateGraph(AgentState)

workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        END: END
    }
)

workflow.add_edge("tools", "agent")


app = workflow.compile()


# =========================
# 🚀 8. Run Pipeline
# =========================
if __name__ == "__main__":
    input_state = {
        "messages": [
            SystemMessage(content="You are a helpful assistant that can use tools."),
            HumanMessage(content="What is 12 * 8? Use tools if needed.")
        ]
    }

    result = app.invoke(input_state)

    print("\n🤖 Final Output:\n")
    print(result["messages"][-1].content)