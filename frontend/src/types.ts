export type DecisionKind = "deterministic" | "dynamic" | "unknown";

export interface GraphNodeData {
  id: string;
  kind: string;
  file: string | null;
  line: number | null;
  subgraph: string | null;
  /** Static branch classification (see _extractor.py): whether a router's
   * routing is a real (LLM-driven) decision or a disguised deterministic
   * if-else. null for non-router nodes. */
  decision_kind: DecisionKind | null;
  /** Short human-readable reason for decision_kind, e.g. "calls llm.invoke()
   * in router". null when decision_kind is null. */
  decision_reason: string | null;
  /** Derived client-side (see graphAnalysis.ts), not part of the server
   * schema: a node with >=2 conditional out-edges. */
  isRouter?: boolean;
  /** Derived client-side: a node with a direct edge to __end__. */
  isTerminal?: boolean;
}

export interface GraphEdgeData {
  src: string;
  tgt: string;
  conditional: boolean;
  label: string | null;
  inferred: boolean;
  /** The source router's decision_kind, propagated onto each of its
   * conditional edges so the UI can style them without re-deriving it.
   * Orthogonal to `inferred` -- an edge can be inferred=false AND
   * branch_kind=deterministic. null for non-conditional edges. */
  branch_kind: DecisionKind | null;
}

export interface GraphData {
  graph_id: string;
  hash: string | null;
  nodes: GraphNodeData[];
  edges: GraphEdgeData[];
  subgraphs: unknown[];
  partial?: boolean;
}

export interface GraphError {
  error: string;
  kind: string;
  detail: Record<string, unknown>;
  traceback?: string;
}

export function isGraphError(x: GraphData | GraphError): x is GraphError {
  return (x as GraphError).error !== undefined;
}
