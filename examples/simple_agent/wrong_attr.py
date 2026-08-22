"""Failure: attr_missing when pointed at the default 'graph' attr, plus
not_a_graph if pointed at 'builder'."""
from typing import TypedDict

from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    text: str


def step(state: State) -> State:
    return {"text": state["text"]}


builder = StateGraph(State)
builder.add_node("step", step)
builder.add_edge(START, "step")
builder.add_edge("step", END)

compiled_graph = builder.compile()  # not named "graph" on purpose
