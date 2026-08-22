"""Failure: factory_needs_args -- callable resolves but requires an argument
that isn't checkpointer-shaped (so the async-CM/checkpointer auto-injection
feature must NOT resolve this one; see orchide_pattern.py for that case)."""
from typing import TypedDict

from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    text: str


def build_graph(db_connection):
    def step(state: State) -> State:
        return {"text": state["text"] + "."}

    builder = StateGraph(State)
    builder.add_node("step", step)
    builder.add_edge(START, "step")
    builder.add_edge("step", END)
    return builder.compile()
