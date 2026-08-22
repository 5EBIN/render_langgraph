"""Happy path: a zero-required-arg factory. Must be found, called (with no
args), and extracted -- not treated as a bare uncompiled StateGraph."""
from typing import TypedDict

from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    text: str


def step(state: State) -> State:
    return {"text": state["text"] + "."}


def build_graph():
    builder = StateGraph(State)
    builder.add_node("step", step)
    builder.add_edge(START, "step")
    builder.add_edge("step", END)
    return builder.compile()
