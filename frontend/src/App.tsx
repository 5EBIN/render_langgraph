import { useCallback, useEffect, useRef, useState } from "react";
import { Background, Controls, MiniMap, ReactFlow, ReactFlowProvider, useReactFlow, type Edge, type Node } from "@xyflow/react";
import DownloadButton from "./DownloadButton";
import RoutedEdge from "./edges/RoutedEdge";
import { layoutGraph } from "./layout";
import ClusterNode from "./nodes/ClusterNode";
import GraphNode from "./nodes/GraphNode";
import type { GraphData, GraphError } from "./types";
import { isGraphError } from "./types";

const nodeTypes = { graphNode: GraphNode, clusterNode: ClusterNode };
const edgeTypes = { routed: RoutedEdge };

interface GraphListEntry {
  id: string;
  label: string;
}

function ErrorPanel({ err }: { err: GraphError }) {
  return (
    <div style={{ padding: 24, fontFamily: "monospace", maxWidth: 720 }}>
      <h2 style={{ color: "#a5333b" }}>render-langgraph: {err.kind}</h2>
      <p>{err.error}</p>
      {err.detail && Object.keys(err.detail).length > 0 && (
        <pre style={{ background: "#0001", padding: 12, borderRadius: 6, overflowX: "auto" }}>
          {JSON.stringify(err.detail, null, 2)}
        </pre>
      )}
      {err.traceback && (
        <details>
          <summary>traceback</summary>
          <pre style={{ overflowX: "auto" }}>{err.traceback}</pre>
        </details>
      )}
    </div>
  );
}

function GraphSelector({
  graphs,
  selected,
  onSelect,
}: {
  graphs: GraphListEntry[];
  selected: string | null;
  onSelect: (id: string) => void;
}) {
  // Only rendered when the project actually has more than one distinct
  // graph -- otherwise this would just be a single, pointless button.
  if (graphs.length <= 1) return null;
  return (
    <div
      style={{
        position: "absolute",
        zIndex: 10,
        top: 8,
        left: 8,
        display: "flex",
        gap: 4,
        background: "var(--legend-bg, #ffffffee)",
        border: "1px solid #0002",
        borderRadius: 8,
        padding: 4,
      }}
    >
      {graphs.map((g) => (
        <button
          key={g.id}
          onClick={() => onSelect(g.id)}
          style={{
            border: "none",
            borderRadius: 6,
            padding: "4px 10px",
            fontSize: 12,
            cursor: "pointer",
            background: g.id === selected ? "#2c7dc9" : "transparent",
            color: g.id === selected ? "#fff" : "inherit",
            fontWeight: g.id === selected ? 600 : 400,
          }}
        >
          {g.label}
        </button>
      ))}
    </div>
  );
}

function Legend({
  hasRealSubgraphs,
  groupRouters,
  onToggleGroup,
}: {
  hasRealSubgraphs: boolean;
  groupRouters: boolean;
  onToggleGroup: (v: boolean) => void;
}) {
  return (
    <div
      style={{
        position: "absolute",
        zIndex: 10,
        bottom: 8,
        left: 8,
        background: "var(--legend-bg, #ffffffee)",
        border: "1px solid #0002",
        borderRadius: 8,
        padding: "8px 12px",
        fontSize: 11,
        lineHeight: 1.6,
        maxWidth: 260,
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
        <span style={{ width: 10, height: 10, background: "#c9791f", clipPath: "polygon(20% 0,80% 0,100% 20%,100% 80%,80% 100%,20% 100%,0 80%,0 20%)" }} />
        router (&ge;2 conditional branches)
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
        <span style={{ width: 10, height: 10, border: "1.5px dashed #3b6ea5", boxSizing: "border-box" }} />
        terminal (edges to __end__)
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
        <span style={{ display: "inline-flex", flexDirection: "column", gap: 1 }}>
          <span style={{ width: 14, height: 1.5, background: "hsl(30, 30%, 36%)" }} />
          <span style={{ width: 14, height: 1.5, background: "hsl(30, 30%, 58%)" }} />
          <span style={{ width: 14, height: 1.5, background: "hsl(30, 30%, 70%)" }} />
        </span>
        back edge (loop, not forward flow) -- shade varies per loop
      </div>
      <div style={{ marginTop: 6, paddingTop: 6, borderTop: "1px solid #0001" }}>
        <div style={{ fontWeight: 600, marginBottom: 2 }}>branch classification</div>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span style={{ display: "inline-flex", alignItems: "center", justifyContent: "center", width: 14, height: 14, borderRadius: "50%", background: "#7a3fd1", color: "#fff", fontSize: 9, fontWeight: 700 }}>
            ⚡
          </span>
          dynamic (real, model-driven decision)
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span style={{ display: "inline-flex", alignItems: "center", justifyContent: "center", width: 14, height: 14, borderRadius: "50%", background: "#5a7a99", color: "#fff", fontSize: 9, fontWeight: 700 }}>
            =
          </span>
          deterministic (disguised if-else)
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span style={{ display: "inline-flex", alignItems: "center", justifyContent: "center", width: 14, height: 14, borderRadius: "50%", background: "#8a8f98", color: "#fff", fontSize: 9, fontWeight: 700 }}>
            ?
          </span>
          unknown (couldn't classify statically)
        </div>
      </div>
      {!hasRealSubgraphs && (
        <label style={{ display: "flex", alignItems: "center", gap: 6, marginTop: 4, cursor: "pointer" }}>
          <input type="checkbox" checked={groupRouters} onChange={(e) => onToggleGroup(e.target.checked)} />
          group workers by nearest router (derived, opt-in)
        </label>
      )}
    </div>
  );
}

/** ReactFlow's `fitView` prop only fits once, at mount -- with the initial
 * empty nodes/edges arrays, since the real graph loads asynchronously after
 * that. Without this, the view stays wherever that first (trivial) fit
 * left it, which can crop nodes clean off-screen once real data arrives.
 * useReactFlow()'s imperative fitView() re-fits whenever the node set
 * actually changes (new graph, grouping toggled, or switching to a
 * different graph in multi-graph mode). */
function FitViewOnChange({ nodes }: { nodes: Node[] }) {
  const { fitView } = useReactFlow();
  useEffect(() => {
    if (nodes.length === 0) return;
    const id = requestAnimationFrame(() => fitView({ padding: 0.15, duration: 200 }));
    return () => cancelAnimationFrame(id);
  }, [nodes, fitView]);
  return null;
}

function GraphCanvas({ nodes, edges, graphId }: { nodes: Node[]; edges: Edge[]; graphId: string }) {
  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      nodeTypes={nodeTypes}
      edgeTypes={edgeTypes}
      proOptions={{ hideAttribution: true }}
      // Default minZoom (0.5) can't zoom out far enough to fit a tall/wide
      // orchestration graph -- fitView silently hits that floor and crops
      // content instead of shrinking further to fit.
      minZoom={0.05}
      maxZoom={2}
    >
      <Background />
      <Controls />
      <MiniMap />
      <FitViewOnChange nodes={nodes} />
      <DownloadButton graphId={graphId} />
    </ReactFlow>
  );
}

export default function App() {
  const [graphData, setGraphData] = useState<GraphData | GraphError | null>(null);
  const [nodes, setNodes] = useState<Node[]>([]);
  const [edges, setEdges] = useState<Edge[]>([]);
  const [groupRouters, setGroupRouters] = useState(false);
  const [graphList, setGraphList] = useState<GraphListEntry[]>([]);
  const [selectedGraphId, setSelectedGraphId] = useState<string | null>(null);
  const lastHash = useRef<string | null>(null);
  const lastGraphData = useRef<GraphData | null>(null);
  const selectedGraphIdRef = useRef<string | null>(null);

  const relayout = useCallback(async (data: GraphData, group: boolean) => {
    const laid = await layoutGraph(data, { groupRouters: group });
    setNodes(laid.nodes);
    setEdges(laid.edges);
  }, []);

  const fetchGraph = useCallback(
    async (graphId?: string, opts?: { force?: boolean }) => {
      const url = graphId ? `/api/graph?graph=${encodeURIComponent(graphId)}` : "/api/graph";
      const res = await fetch(url);
      const data = (await res.json()) as GraphData | GraphError;
      setGraphData(data);
      if (!isGraphError(data)) {
        lastGraphData.current = data;
        if (opts?.force || data.hash !== lastHash.current) {
          lastHash.current = data.hash;
          await relayout(data, groupRouters);
        }
      }
    },
    [relayout, groupRouters]
  );

  const fetchGraphList = useCallback(async () => {
    try {
      const res = await fetch("/api/graphs");
      if (!res.ok) return;
      const body = (await res.json()) as { graphs?: GraphListEntry[] };
      const graphs = body.graphs ?? [];
      setGraphList(graphs);
      if (graphs.length > 0 && !selectedGraphIdRef.current) {
        selectedGraphIdRef.current = graphs[0].id;
        setSelectedGraphId(graphs[0].id);
      }
    } catch {
      // /api/graphs not reachable -- fall back silently to the server's
      // single default graph, same as before this endpoint existed.
    }
  }, []);

  useEffect(() => {
    (async () => {
      await fetchGraphList();
      await fetchGraph(selectedGraphIdRef.current ?? undefined);
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const es = new EventSource("/api/events");
    es.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        if (msg.changed) {
          fetchGraphList();
          fetchGraph(selectedGraphIdRef.current ?? undefined);
        }
      } catch {
        // ignore malformed/keepalive frames
      }
    };
    es.onerror = () => {
      // browser auto-reconnects EventSource; nothing to do
    };
    return () => es.close();
  }, [fetchGraph, fetchGraphList]);

  const handleToggleGroup = (value: boolean) => {
    setGroupRouters(value);
    if (lastGraphData.current) {
      relayout(lastGraphData.current, value);
    }
  };

  const handleSelectGraph = (id: string) => {
    selectedGraphIdRef.current = id;
    setSelectedGraphId(id);
    // Switching to a different graph must always relayout, even if its hash
    // happens to equal whatever was last shown -- the hash-skip check
    // exists to preserve pan/zoom across unchanged file-watch reloads of
    // the SAME graph, not to decide whether a different graph needs drawing.
    fetchGraph(id, { force: true });
  };

  if (!graphData) {
    return <div style={{ padding: 24, fontFamily: "sans-serif" }}>loading…</div>;
  }

  if (isGraphError(graphData)) {
    return (
      <>
        <GraphSelector graphs={graphList} selected={selectedGraphId} onSelect={handleSelectGraph} />
        <ErrorPanel err={graphData} />
      </>
    );
  }

  const hasRealSubgraphs = graphData.nodes.some((n) => n.subgraph);

  return (
    <div style={{ width: "100vw", height: "100vh" }}>
      {graphData.partial && (
        <div style={{ position: "absolute", zIndex: 10, top: 8, left: graphList.length > 1 ? 220 : 8, background: "#c98a2c", color: "#fff", padding: "4px 10px", borderRadius: 6, fontSize: 12 }}>
          partial result — some edges couldn't be resolved statically
        </div>
      )}
      <ReactFlowProvider>
        <GraphCanvas nodes={nodes} edges={edges} graphId={selectedGraphId || graphData.graph_id} />
      </ReactFlowProvider>
      <GraphSelector graphs={graphList} selected={selectedGraphId} onSelect={handleSelectGraph} />
      <Legend hasRealSubgraphs={hasRealSubgraphs} groupRouters={groupRouters} onToggleGroup={handleToggleGroup} />
    </div>
  );
}
