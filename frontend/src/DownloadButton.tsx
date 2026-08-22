import { useState } from "react";
import { getNodesBounds, getViewportForBounds, useReactFlow } from "@xyflow/react";
import { toSvg } from "html-to-image";

const MAX_DIMENSION = 2400;
const PADDING = 0.1;

function triggerDownload(dataUrl: string, filename: string) {
  const a = document.createElement("a");
  a.setAttribute("download", filename);
  a.setAttribute("href", dataUrl);
  a.click();
}

export default function DownloadButton({ graphId }: { graphId: string }) {
  const { getNodes } = useReactFlow();
  const [busy, setBusy] = useState(false);

  const handleClick = async () => {
    const viewportEl = document.querySelector(".react-flow__viewport") as HTMLElement | null;
    if (!viewportEl) return;

    setBusy(true);
    try {
      const bounds = getNodesBounds(getNodes());
      // Size the export canvas to the graph's own aspect ratio (our graphs
      // range from tall linear chains to very wide multi-router layouts) so
      // it isn't stretched/cropped into a fixed portrait or landscape frame.
      let width = MAX_DIMENSION;
      let height = Math.round((bounds.height / bounds.width) * MAX_DIMENSION);
      if (height > MAX_DIMENSION) {
        height = MAX_DIMENSION;
        width = Math.round((bounds.width / bounds.height) * MAX_DIMENSION);
      }

      const viewport = getViewportForBounds(bounds, width, height, 0.05, 4, PADDING);

      const dataUrl = await toSvg(viewportEl, {
        backgroundColor: "#ffffff",
        width,
        height,
        style: {
          width: String(width),
          height: String(height),
          transform: `translate(${viewport.x}px, ${viewport.y}px) scale(${viewport.zoom})`,
        },
      });
      triggerDownload(dataUrl, `${graphId || "graph"}.svg`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <button
      onClick={handleClick}
      disabled={busy}
      style={{
        position: "absolute",
        zIndex: 10,
        top: 8,
        right: 8,
        background: "var(--legend-bg, #ffffffee)",
        border: "1px solid #0002",
        borderRadius: 8,
        padding: "6px 12px",
        fontSize: 12,
        cursor: busy ? "default" : "pointer",
        opacity: busy ? 0.6 : 1,
      }}
    >
      {busy ? "exporting…" : "⬇ download SVG"}
    </button>
  );
}
