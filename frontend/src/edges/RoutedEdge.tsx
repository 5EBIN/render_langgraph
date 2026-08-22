import { BaseEdge, EdgeLabelRenderer, getBezierPath, type EdgeProps } from "@xyflow/react";
import type { ElkSection, RoutedElkEdge } from "../layout";

/**
 * React Flow's default edge rendering ignores whatever ELK actually
 * computed -- it just draws its own bezier between node anchor points. To
 * make P4 (orthogonal routing / merged fan-out) and P5 (label placement
 * allocated space, not a post-hoc midpoint guess) visible at all, this
 * component reads the real ELK edge section/label geometry we stashed in
 * `data.elk` and draws that instead.
 *
 * Back edges are the one deliberate exception: rendering a reversed cyclic
 * edge with rigid orthogonal bend points tends to read as a confusing
 * zig-zag back up the canvas. Since the point of marking back edges is
 * precisely to make loops read as loops rather than forward flow, they're
 * drawn as a soft bezier curve instead (see P1).
 */
function buildOrthogonalPath(
  sections: ElkSection[] | undefined,
  fallback: { sourceX: number; sourceY: number; targetX: number; targetY: number }
): string {
  if (!sections || sections.length === 0) {
    const { sourceX, sourceY, targetX, targetY } = fallback;
    return `M ${sourceX},${sourceY} L ${targetX},${targetY}`;
  }
  const segments: string[] = [];
  for (const section of sections) {
    const points = [section.startPoint, ...(section.bendPoints ?? []), section.endPoint];
    segments.push(`M ${points[0].x},${points[0].y}`);
    for (let i = 1; i < points.length; i++) {
      segments.push(`L ${points[i].x},${points[i].y}`);
    }
  }
  return segments.join(" ");
}

export default function RoutedEdge({
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  style,
  markerEnd,
  data,
  label,
  labelStyle,
}: EdgeProps) {
  const elkEdge = (data as { elk?: RoutedElkEdge } | undefined)?.elk;
  const isBackEdge = Boolean((data as { isBackEdge?: boolean } | undefined)?.isBackEdge);

  let path: string;
  let labelX: number;
  let labelY: number;

  if (isBackEdge) {
    const [bezierPath, bx, by] = getBezierPath({
      sourceX,
      sourceY,
      targetX,
      targetY,
      sourcePosition,
      targetPosition,
      curvature: 0.6,
    });
    path = bezierPath;
    labelX = bx;
    labelY = by;
  } else {
    path = buildOrthogonalPath(elkEdge?.sections, { sourceX, sourceY, targetX, targetY });
    const elkLabel = elkEdge?.labels?.[0];
    labelX = elkLabel ? elkLabel.x + elkLabel.width / 2 : (sourceX + targetX) / 2;
    labelY = elkLabel ? elkLabel.y + elkLabel.height / 2 : (sourceY + targetY) / 2;
  }

  return (
    <>
      <BaseEdge path={path} style={style} markerEnd={markerEnd} />
      {label && (
        <EdgeLabelRenderer>
          <div
            style={{
              position: "absolute",
              transform: `translate(-50%, -50%) translate(${labelX}px, ${labelY}px)`,
              background: "var(--edge-label-bg, #ffffffee)",
              padding: "1px 5px",
              borderRadius: 4,
              fontSize: 11,
              pointerEvents: "none",
              whiteSpace: "nowrap",
              ...labelStyle,
            }}
          >
            {label}
          </div>
        </EdgeLabelRenderer>
      )}
    </>
  );
}
