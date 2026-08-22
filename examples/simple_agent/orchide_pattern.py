"""Fixture mirroring a common production pattern (seen in the "Orchide"
report that motivated this feature): .compile() lives in a private function
requiring a checkpointer, wrapped by an async context manager for
production use, with a sync build_* helper for tests/dev. Two independent
compiled graphs (planning + code) exercise discover.py's multi-graph
listing and render_langgraph's async-CM/checkpointer auto-resolution.
"""
from contextlib import asynccontextmanager
from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    text: str


def _step(name: str):
    def fn(state: State) -> State:
        return {"text": state["text"] + f".{name}"}

    return fn


# --- graph 1: planning -- checkpointer-only, fully auto-resolvable either
# through the async CM (planning_graph) or the sync helper (build_planning_graph) ---
def _compile_planning_graph(checkpointer):
    builder = StateGraph(State)
    builder.add_node("plan_project", _step("plan_project"))
    builder.add_node("design_node", _step("design_node"))
    builder.add_edge(START, "plan_project")
    builder.add_edge("plan_project", "design_node")
    builder.add_edge("design_node", END)
    return builder.compile(checkpointer=checkpointer)


@asynccontextmanager
async def planning_graph():
    checkpointer = InMemorySaver()
    yield _compile_planning_graph(checkpointer)


def build_planning_graph(checkpointer=None):
    return _compile_planning_graph(checkpointer or InMemorySaver())


# --- graph 2: code-gen -- checkpointer AND a second required, non-checkpointer
# arg (llm_client). Going through the async CM, only checkpointer can be
# auto-resolved; llm_client must still be reported. The sync helper defaults
# both, so it resolves fully on its own. ---
def _compile_code_graph(checkpointer, llm_client):
    builder = StateGraph(State)
    builder.add_node("generate", _step("generate"))
    builder.add_edge(START, "generate")
    builder.add_edge("generate", END)
    return builder.compile(checkpointer=checkpointer)


@asynccontextmanager
async def code_graph():
    checkpointer = InMemorySaver()
    yield _compile_code_graph(checkpointer, "prod-llm-client")


def build_code_graph(checkpointer=None, llm_client=None):
    return _compile_code_graph(checkpointer or InMemorySaver(), llm_client)
