"""Happy path: simple linear graph, no branching."""
from typing import TypedDict

from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    text: str


def upper(state: State) -> State:
    return {"text": state["text"].upper()}


def exclaim(state: State) -> State:
    return {"text": state["text"] + "!"}


builder = StateGraph(State)
builder.add_node("upper", upper)
builder.add_node("exclaim", exclaim)
builder.add_edge(START, "upper")
builder.add_edge("upper", "exclaim")
builder.add_edge("exclaim", END)

graph = builder.compile()
