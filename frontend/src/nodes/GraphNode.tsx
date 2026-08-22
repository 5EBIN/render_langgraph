import { Handle, Position, type NodeProps } from "@xyflow/react";
import type { GraphNodeData } from "../types";

const KIND_COLOR: Record<string, string> = {
  function: "#3b6ea5",
  tool: "#3f9142",
  subgraph: "#8a4fd1",
  runnable: "#5a5f66",
  start: "#1f8a4c",
  end: "#a5333b",
};

const ROUTER_COLOR = "#c9791f";

// Cuts all four corners, giving router nodes a distinct octagon silhouette
// so they read as a different kind of thing at a glance, not just a
// different color of the same rectangle every other node uses.
const OCTAGON_CLIP =
  "polygon(12px 0, calc(100% - 12px) 0, 100% 12px, 100% calc(100% - 12px), calc(100% - 12px) 100%, 12px 100%, 0 calc(100% - 12px), 0 12px)";

// render-langgraph's signature feature: is this router a real (LLM-driven)
// decision, or a disguised deterministic if-else? Badge + tooltip only --
// the classification itself comes from the backend (_extractor.py) or
// static_parse.py, never re-derived here.
const DECISION_BADGE: Record<string, { symbol: string; color: string; label: string }> = {
  dynamic: { symbol: "⚡", color: "#7a3fd1", label: "dynamic (model-driven) decision" },
  deterministic: { symbol: "=", color: "#5a7a99", label: "deterministic (if-else) routing" },
  unknown: { symbol: "?", color: "#8a8f98", label: "routing kind unknown (couldn't classify statically)" },
};

function toVscodeUri(file: string, line: number | null): string {
  let p = file.replace(/\\/g, "/");
  if (!p.startsWith("/")) p = "/" + p;
  return `vscode://file${p}${line ? ":" + line : ""}`;
}

export default function GraphNode({ data }: NodeProps) {
  const node = data as unknown as GraphNodeData;
  const isStartOrEnd = node.kind === "start" || node.kind === "end";
  const isRouter = Boolean(node.isRouter) && !isStartOrEnd;
  const isTerminal = Boolean(node.isTerminal) && !isStartOrEnd && !isRouter;
  const color = isRouter ? ROUTER_COLOR : KIND_COLOR[node.kind] ?? "#5a5f66";
  const decision = node.decision_kind ? DECISION_BADGE[node.decision_kind] : null;

  const handleClick = () => {
    if (!node.file) return;
    const uri = toVscodeUri(node.file, node.line);
    window.location.href = uri;
    navigator.clipboard?.writeText(`${node.file}:${node.line ?? ""}`).catch(() => {});
  };

  const titleParts = [node.file ? `${node.file}:${node.line ?? "?"}` : "no source location"];
  if (decision) titleParts.push(`${decision.label}${node.decision_reason ? ` -- ${node.decision_reason}` : ""}`);

  return (
    <div
      onClick={handleClick}
      title={titleParts.join("\n")}
      style={{
        border: `1.5px ${isTerminal ? "dashed" : "solid"} ${color}`,
        borderLeft: isRouter ? `1.5px solid ${color}` : `6px solid ${color}`,
        borderRadius: isRouter ? 0 : 8,
        clipPath: isRouter ? OCTAGON_CLIP : undefined,
        background: isRouter ? "rgba(201,121,31,0.08)" : "var(--node-bg, #fff)",
        padding: isRouter ? "8px 16px" : "8px 12px",
        width: "100%",
        height: "100%",
        boxSizing: "border-box",
        cursor: node.file ? "pointer" : "default",
        display: "flex",
        flexDirection: "column",
        justifyContent: "center",
        fontSize: 13,
        position: "relative",
      }}
    >
      <Handle type="target" position={Position.Left} />
      {decision && (
        <span
          style={{
            position: "absolute",
            top: -9,
            right: -9,
            width: 18,
            height: 18,
            borderRadius: "50%",
            background: decision.color,
            color: "#fff",
            fontSize: 11,
            lineHeight: "18px",
            textAlign: "center",
            fontWeight: 700,
            boxShadow: "0 0 0 2px var(--node-bg, #fff)",
          }}
        >
          {decision.symbol}
        </span>
      )}
      <div style={{ fontWeight: 600, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
        {node.id}
      </div>
      <div style={{ fontSize: 10, color, textTransform: "uppercase", letterSpacing: 0.5 }}>
        {isRouter ? "router" : node.kind}
        {isTerminal && " · terminal"}
      </div>
      <Handle type="source" position={Position.Right} />
    </div>
  );
}
