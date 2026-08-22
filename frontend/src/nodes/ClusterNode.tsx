import type { NodeProps } from "@xyflow/react";

interface ClusterNodeData {
  label: string;
  /** true when this cluster came from the opt-in derived router-ownership
   * heuristic (Case B) rather than a real compiled subgraph (Case A) --
   * labeled differently so it's never mistaken for actual graph structure. */
  derived: boolean;
}

export default function ClusterNode({ data }: NodeProps) {
  const { label, derived } = data as unknown as ClusterNodeData;
  return (
    <div
      style={{
        width: "100%",
        height: "100%",
        border: `1.5px dashed ${derived ? "#9a7fd1" : "#8a4fd1"}`,
        borderRadius: 12,
        background: derived ? "rgba(154,127,209,0.05)" : "rgba(138,79,209,0.06)",
        boxSizing: "border-box",
        position: "relative",
        pointerEvents: "none",
      }}
    >
      <div
        style={{
          position: "absolute",
          top: -11,
          left: 12,
          fontSize: 11,
          fontWeight: 600,
          color: derived ? "#9a7fd1" : "#8a4fd1",
          background: "var(--cluster-label-bg, #fff)",
          padding: "0 6px",
          borderRadius: 4,
        }}
      >
        {label}
        {derived && <span style={{ fontWeight: 400, opacity: 0.8 }}> (grouped)</span>}
      </div>
    </div>
  );
}
