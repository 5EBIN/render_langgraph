import ELK, { ElkExtendedEdge, ElkNode } from "elkjs/lib/elk.bundled.js";
import { MarkerType, type Edge, type Node } from "@xyflow/react";
import { classifyBackEdges, computeNodeRoles, computeRouterClusters } from "./graphAnalysis";
import type { GraphData, GraphEdgeData } from "./types";

const elk = new ELK();

const NODE_WIDTH = 180;
const NODE_HEIGHT = 56;
const CLUSTER_PREFIX = "cluster:";

interface EdgeVisual {
  stroke: string;
  strokeDasharray: string | undefined;
  strokeWidth: number;
  opacity: number;
}

// Multiple back edges routinely converge into the same corner of a
// left-to-right layout (e.g. several router fallbacks all looping back to
// a shared coordinator node) -- ELK's orthogonal routing sends each on a
// long swing around the whole diagram, so with one flat color they render
// as an indistinguishable tangle. Cycling lightness within the same warm
// hue keeps the "this is a loop, not forward flow" identity intact (still
// dotted, still clearly not one of the blue/purple/grey branch-kind
// colors) while letting the eye trace an individual loop through the mess.
// The steps are interleaved (not a simple light->dark ramp) so back edges
// with adjacent indices -- often adjacent in the drawing too, since they
// tend to come from siblings in the edge list -- contrast rather than
// blend into their neighbor.
const BACK_EDGE_HUE = 30; // matches the original flat #9a8f7a tan
const BACK_EDGE_LIGHTNESS_STEPS = [58, 36, 70, 26, 48, 62, 32];
function backEdgeStroke(variant: number): string {
  const lightness = BACK_EDGE_LIGHTNESS_STEPS[variant % BACK_EDGE_LIGHTNESS_STEPS.length];
  return `hsl(${BACK_EDGE_HUE}, 30%, ${lightness}%)`;
}

// Branch classification is the primary signal for a conditional edge's
// look: dynamic (real, model-driven decision) = solid accent; deterministic
// (disguised if-else) = subtle/dashed; unknown = dotted grey. Back edges
// (loops) keep their own curved/lighter treatment regardless -- that's
// about structure, not decision kind, and takes priority since a back edge
// reads as "not forward flow" first.
function _edgeVisual(e: GraphEdgeData, isBackEdge: boolean, backEdgeVariant: number): EdgeVisual {
  if (isBackEdge) {
    return { stroke: backEdgeStroke(backEdgeVariant), strokeDasharray: "2 4", strokeWidth: 1.5, opacity: 0.85 };
  }
  if (!e.conditional) {
    return { stroke: "#8a8f98", strokeDasharray: undefined, strokeWidth: 1.5, opacity: 1 };
  }
  switch (e.branch_kind) {
    case "dynamic":
      return { stroke: "#7a3fd1", strokeDasharray: undefined, strokeWidth: 2.5, opacity: 1 };
    case "deterministic":
      return { stroke: "#5a7a99", strokeDasharray: "5 3", strokeWidth: 1.75, opacity: 0.9 };
    default:
      return { stroke: "#9a9a9a", strokeDasharray: "1 3", strokeWidth: 1.5, opacity: 0.8 };
  }
}

// P1: RIGHT (left-to-right) direction -- chosen over DOWN so long chains
// (the common case: mostly-linear orchestration graphs with a handful of
// branch clusters) use a landscape viewport's width instead of running off
// the bottom of a tall, narrow column. __start__/__end__ anchoring
// (layerConstraint FIRST_SEPARATE/LAST_SEPARATE, set per-node below) is
// direction-relative in ELK, so this makes __start__ leftmost and __end__
// rightmost rather than top/bottom -- that's the deliberate trade being
// made here, not a leftover from the old top-down layout.
// Also: a deterministic cycle-breaking strategy so back edges are chosen
// from the entry point instead of letting an arbitrary cycle node get
// hoisted before __start__.
// P4: orthogonal routing + wider spacing + edge merging to tame router
// fan-out spaghetti.
// P5: edge labels are passed to ELK as label elements (see buildElkGraph)
// and elk.edgeLabels.placement=CENTER tells ELK to allocate real space for
// them during layout, rather than us drawing them post-hoc at the polyline
// midpoint where they'd collide with crossing edges.
// hierarchyHandling is harmless to set even when there's no nesting (flat
// graphs simply have no children to recurse into), so it's always on
// rather than conditionally wired.
const ROOT_LAYOUT_OPTIONS: Record<string, string> = {
  "elk.algorithm": "layered",
  "elk.direction": "RIGHT",
  // Perpendicular-to-flow spacing (vertical, since direction is RIGHT):
  // widened from the original 56 -- with several sibling nodes stacked in
  // the same layer, tight spacing left their connecting edges running
  // close enough together to visually merge into each other.
  "elk.spacing.nodeNode": "110",
  "elk.layered.spacing.nodeNodeBetweenLayers": "110",
  "elk.spacing.edgeLabel": "18",
  "elk.layered.spacing.edgeNodeBetweenLayers": "40",
  // Explicit clearance between parallel edge segments -- without this ELK
  // will happily route two edges close enough to overlap visually,
  // especially where several router branches fan out from the same node.
  "elk.spacing.edgeEdge": "32",
  "elk.layered.spacing.edgeEdgeBetweenLayers": "32",
  "elk.layered.cycleBreaking.strategy": "DEPTH_FIRST",
  "elk.edgeRouting": "ORTHOGONAL",
  "elk.layered.mergeEdges": "true",
  "elk.edgeLabels.placement": "CENTER",
  "elk.hierarchyHandling": "INCLUDE_CHILDREN",
};

const CLUSTER_LAYOUT_OPTIONS: Record<string, string> = {
  "elk.padding": "[top=36,left=18,bottom=18,right=18]",
};

export interface LayoutOptions {
  /** Case B opt-in: derive router-ownership clusters when the graph has no
   * real compiled subgraphs. Off by default -- never silently regroups a
   * user's graph (see graphAnalysis.computeRouterClusters). Ignored when
   * the graph DOES have real subgraph data (Case A always applies, since
   * that reflects actual structure rather than a guess). */
  groupRouters?: boolean;
}

export interface LayoutResult {
  nodes: Node[];
  edges: Edge[];
}

const layoutCache = new Map<string, LayoutResult>();

function cacheKey(hash: string, groupRouters: boolean): string {
  return `${hash}::${groupRouters ? "grouped" : "flat"}`;
}

interface ElkPoint {
  x: number;
  y: number;
}
interface ElkLabel extends ElkPoint {
  width: number;
  height: number;
  text?: string;
}
export interface ElkSection {
  startPoint: ElkPoint;
  endPoint: ElkPoint;
  bendPoints?: ElkPoint[];
}
export interface RoutedElkEdge {
  sections?: ElkSection[];
  labels?: ElkLabel[];
}

function buildElkGraph(data: GraphData, groupOf: Map<string, string>): ElkNode {
  const clusterIds = Array.from(new Set(groupOf.values()));
  const rootChildren: ElkNode[] = [];
  const clusterChildren = new Map<string, ElkNode[]>();

  for (const n of data.nodes) {
    const layoutOptions: Record<string, string> = {};
    if (n.id === "__start__") layoutOptions["elk.layered.layering.layerConstraint"] = "FIRST_SEPARATE";
    if (n.id === "__end__") layoutOptions["elk.layered.layering.layerConstraint"] = "LAST_SEPARATE";

    const elkNode: ElkNode = {
      id: n.id,
      width: NODE_WIDTH,
      height: NODE_HEIGHT,
      ...(Object.keys(layoutOptions).length ? { layoutOptions } : {}),
    };

    const cluster = groupOf.get(n.id);
    if (cluster) {
      if (!clusterChildren.has(cluster)) clusterChildren.set(cluster, []);
      clusterChildren.get(cluster)!.push(elkNode);
    } else {
      rootChildren.push(elkNode);
    }
  }

  const clusterEdges = new Map<string, ElkExtendedEdge[]>();
  const rootEdges: ElkExtendedEdge[] = [];

  data.edges.forEach((e, i) => {
    const elkEdge: ElkExtendedEdge = {
      id: `e${i}-${e.src}-${e.tgt}`,
      sources: [e.src],
      targets: [e.tgt],
      labels: e.label ? [{ text: e.label, width: e.label.length * 7 + 8, height: 14 }] : [],
    };
    const srcCluster = groupOf.get(e.src);
    const tgtCluster = groupOf.get(e.tgt);
    // Each edge is declared at the lowest container that contains both its
    // endpoints. Our clustering is only ever one level deep, so that's
    // either "inside a single cluster" or "at the root" -- root also
    // correctly covers cross-cluster and cluster<->top-level edges, which
    // ELK routes as hierarchy-crossing edges from their common ancestor.
    if (srcCluster && srcCluster === tgtCluster) {
      if (!clusterEdges.has(srcCluster)) clusterEdges.set(srcCluster, []);
      clusterEdges.get(srcCluster)!.push(elkEdge);
    } else {
      rootEdges.push(elkEdge);
    }
  });

  for (const clusterId of clusterIds) {
    rootChildren.push({
      id: CLUSTER_PREFIX + clusterId,
      layoutOptions: CLUSTER_LAYOUT_OPTIONS,
      children: clusterChildren.get(clusterId) ?? [],
      edges: clusterEdges.get(clusterId) ?? [],
    });
  }

  return {
    id: "root",
    layoutOptions: ROOT_LAYOUT_OPTIONS,
    children: rootChildren,
    edges: rootEdges,
  };
}

interface PositionedNode {
  id: string;
  x: number;
  y: number;
  width: number;
  height: number;
  isCluster: boolean;
  parentClusterId?: string;
}

function extractPositions(elkNode: ElkNode, parentClusterId?: string): PositionedNode[] {
  const out: PositionedNode[] = [];
  for (const child of elkNode.children ?? []) {
    const isCluster = child.id.startsWith(CLUSTER_PREFIX);
    out.push({
      id: child.id,
      x: child.x ?? 0,
      y: child.y ?? 0,
      width: child.width ?? NODE_WIDTH,
      height: child.height ?? NODE_HEIGHT,
      isCluster,
      parentClusterId,
    });
    if (child.children?.length) {
      out.push(...extractPositions(child, child.id));
    }
  }
  return out;
}

function collectElkEdges(elkNode: ElkNode): Map<string, RoutedElkEdge> {
  const out = new Map<string, RoutedElkEdge>();
  for (const e of elkNode.edges ?? []) {
    out.set(e.id!, e as RoutedElkEdge);
  }
  for (const child of elkNode.children ?? []) {
    for (const [id, e] of collectElkEdges(child)) out.set(id, e);
  }
  return out;
}

export async function layoutGraph(data: GraphData, options: LayoutOptions = {}): Promise<LayoutResult> {
  const groupRouters = options.groupRouters ?? false;
  if (data.hash) {
    const cached = layoutCache.get(cacheKey(data.hash, groupRouters));
    if (cached) return cached;
  }

  const nodeIds = data.nodes.map((n) => n.id);
  const { isRouter, isTerminal } = computeNodeRoles(nodeIds, data.edges);
  const backEdgeIndices = classifyBackEdges(nodeIds, data.edges);

  // Case A (real subgraphs) always applies -- it's actual structure, not a
  // guess. Case B (derived router clusters) only kicks in when there's no
  // real subgraph data AND the caller opted in.
  const hasRealSubgraphs = data.nodes.some((n) => n.subgraph);
  const routerClusters = !hasRealSubgraphs && groupRouters ? computeRouterClusters(nodeIds, data.edges, isRouter) : new Map<string, string>();

  const groupOf = new Map<string, string>();
  if (hasRealSubgraphs) {
    for (const n of data.nodes) {
      if (n.subgraph) groupOf.set(n.id, n.subgraph);
    }
  } else {
    for (const [node, router] of routerClusters) groupOf.set(node, router);
  }

  const elkGraph = buildElkGraph(data, groupOf);
  const layouted = await elk.layout(elkGraph);

  const positioned = extractPositions(layouted);
  const positionById = new Map(positioned.map((p) => [p.id, p]));
  const nodeById = new Map(data.nodes.map((n) => [n.id, n]));

  const nodes: Node[] = [];
  // Cluster containers must come before their children in React Flow's
  // nodes array.
  for (const p of positioned.filter((p) => p.isCluster)) {
    const clusterId = p.id.slice(CLUSTER_PREFIX.length);
    nodes.push({
      id: p.id,
      type: "clusterNode",
      position: { x: p.x, y: p.y },
      data: { label: clusterId, derived: !hasRealSubgraphs },
      style: { width: p.width, height: p.height },
      selectable: false,
      draggable: false,
      zIndex: -1,
    });
  }
  for (const p of positioned.filter((p) => !p.isCluster)) {
    const original = nodeById.get(p.id);
    if (!original) continue;
    nodes.push({
      id: p.id,
      type: "graphNode",
      position: { x: p.x, y: p.y },
      data: {
        ...original,
        isRouter: isRouter.get(p.id) ?? false,
        isTerminal: isTerminal.get(p.id) ?? false,
      },
      width: NODE_WIDTH,
      height: NODE_HEIGHT,
      ...(p.parentClusterId ? { parentId: p.parentClusterId, extent: "parent" as const } : {}),
    });
  }

  const elkEdgesById = collectElkEdges(layouted);
  let backEdgeVariant = 0;
  const edges: Edge[] = data.edges.map((e, i) => {
    const id = `e${i}-${e.src}-${e.tgt}`;
    const isBackEdge = backEdgeIndices.has(i);
    const visual = _edgeVisual(e, isBackEdge, isBackEdge ? backEdgeVariant++ : 0);
    return {
      id,
      source: e.src,
      target: e.tgt,
      type: "routed",
      label: e.label ?? undefined,
      animated: false,
      style: {
        stroke: visual.stroke,
        strokeDasharray: visual.strokeDasharray,
        strokeWidth: visual.strokeWidth,
        opacity: visual.opacity,
      },
      markerEnd: { type: MarkerType.ArrowClosed, width: 18, height: 18, color: visual.stroke },
      labelStyle: { fill: visual.stroke, fontSize: 11 },
      data: {
        conditional: e.conditional,
        inferred: e.inferred,
        branchKind: e.branch_kind,
        isBackEdge,
        elk: elkEdgesById.get(id),
      },
      zIndex: isBackEdge ? 0 : 1,
    };
  });

  const result = { nodes, edges };
  if (data.hash) layoutCache.set(cacheKey(data.hash, groupRouters), result);
  return result;
}
