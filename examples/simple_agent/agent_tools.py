"""Happy path: agent+tools loop with a conditional router using a Literal
return annotation, so langgraphv can narrow the branch instead of fanning
out to every node."""
from typing import Literal, TypedDict


from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    messages: list[str]
    steps: int


def agent(state: State) -> State:
    return {"messages": state["messages"] + ["agent thinking"], "steps": state["steps"] + 1}


def tools(state: State) -> State:
    return {"messages": state["messages"] + ["tool result"]}


def route(state: State) -> Literal["tools", "__end__"]:
    if state["steps"] >= 2:
        return "__end__"
    return "tools"


builder = StateGraph(State)
builder.add_node("agent", agent)
builder.add_node("tools", tools)
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", route)
builder.add_edge("tools", "agent")

graph = builder.compile()
