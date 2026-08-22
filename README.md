# render-langgraph

A zero-config CLI that renders a [LangGraph](https://github.com/langchain-ai/langgraph)
graph in the browser -- visualize the nodes and edges of any compiled LangGraph graph
without Docker, without running LangGraph Studio, and without executing your agent.

```bash
pip install render-langgraph
cd my-agent-project
render-langgraph          # opens browser, graph rendered
```

## What it does

`render-langgraph` finds your compiled graph (via `langgraph.json`, an AST scan, or an
explicit `path/to/file.py:attr`), spawns your project's own Python interpreter
to call `get_graph(xray=True)` on it, and renders the result with React Flow +
elkjs. No Docker, no execution beyond importing your module, no state inspection --
it reads the graph structure once and shows it to you.

## Usage

```bash
render-langgraph                          # auto-discover
render-langgraph src/agent/graph.py:graph # explicit target
render-langgraph --graph my_graph         # pick an entry from langgraph.json
render-langgraph --static path.py         # AST-only, never imports
render-langgraph --json                   # print graph JSON, no server
```

## Status

v0.1 walking skeleton: runtime extraction, discovery cascade, static AST
fallback, and a served React Flow viewer with live reload on save.

## License

MIT
