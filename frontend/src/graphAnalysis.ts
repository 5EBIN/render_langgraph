/**
 * Pure graph-algorithm helpers over the flat node/edge JSON the server
 * already sends. Nothing here needs new backend fields -- role and
 * back-edge classification only need adjacency, which the existing
 * GraphEdgeData[] already gives us; adding them would be schema churn for
 * derivable data.
 */
import type { GraphEdgeData } from "./types";

export interface NodeRoles {
  isRouter: Map<string, boolean>;
  isTerminal: Map<string, boolean>;
}

/** A router is a node with >=2 conditional out-edges (a real branch point,
 * not just a single declared-conditional passthrough). A terminal is a
 * node with a direct edge to __end__. */
export function computeNodeRoles(nodeIds: string[], edges: GraphEdgeData[]): NodeRoles {
  const conditionalOutCount = new Map<string, number>();
  const endTargets = new Set<string>();

  for (const e of edges) {
    if (e.conditional) {
      conditionalOutCount.set(e.src, (conditionalOutCount.get(e.src) ?? 0) + 1);
    }
    if (e.tgt === "__end__") {
      endTargets.add(e.src);
    }
  }

  const isRouter = new Map<string, boolean>();
  const isTerminal = new Map<string, boolean>();
  for (const id of nodeIds) {
    isRouter.set(id, (conditionalOutCount.get(id) ?? 0) >= 2);
    isTerminal.set(id, endTargets.has(id));
  }
  return { isRouter, isTerminal };
}

/**
 * Classic DFS back-edge classification for a directed graph: an edge (u, v)
 * is a back edge if v is an ancestor of u in the DFS tree (i.e. still on
 * the recursion stack when u is visited). This is computed independently
 * of whatever cycle-breaking ELK does internally for layering -- ELK's
 * layouted output doesn't expose which edges it reversed, and we want a
 * deterministic, explainable "this edge is a loop" signal for rendering
 * regardless of layout internals.
 *
 * Iterative (explicit stack) to avoid recursion-depth concerns on large
 * graphs. DFS starts from __start__ when present (matching the graph's
 * real entry point) so the classification lines up with "flow forward
 * from the top", then covers any remaining unvisited nodes.
 *
 * Returns the set of edge indices (position in the `edges` array) that are
 * back edges.
 */
export function classifyBackEdges(nodeIds: string[], edges: GraphEdgeData[]): Set<number> {
  const adjacency = new Map<string, { tgt: string; idx: number }[]>();
  edges.forEach((e, idx) => {
    if (!adjacency.has(e.src)) adjacency.set(e.src, []);
    adjacency.get(e.src)!.push({ tgt: e.tgt, idx });
  });

  const visited = new Set<string>();
  const onStack = new Set<string>();
  const backEdges = new Set<number>();

  // Explicit-stack DFS with per-node child-iteration cursors so we can
  // pop/push without recursion.
  function dfsFrom(root: string) {
    const stack: { node: string; children: { tgt: string; idx: number }[]; i: number }[] = [];
    const push = (node: string) => {
      visited.add(node);
      onStack.add(node);
      stack.push({ node, children: adjacency.get(node) ?? [], i: 0 });
    };

    push(root);
    while (stack.length) {
      const frame = stack[stack.length - 1];
      if (frame.i >= frame.children.length) {
        onStack.delete(frame.node);
        stack.pop();
        continue;
      }
      const { tgt, idx } = frame.children[frame.i];
      frame.i++;
      if (onStack.has(tgt)) {
        backEdges.add(idx);
      } else if (!visited.has(tgt)) {
        push(tgt);
      }
    }
  }

  const order = nodeIds.includes("__start__")
    ? ["__start__", ...nodeIds.filter((n) => n !== "__start__")]
    : nodeIds;
  for (const id of order) {
    if (!visited.has(id)) dfsFrom(id);
  }
  return backEdges;
}

/**
 * Opt-in, heuristic router-ownership clustering for graphs with no real
 * compiled subgraphs (see layout.ts Case B). "Exclusively reachable from a
 * given router" is interpreted as: multi-source BFS from every router
 * simultaneously (not crossing through *other* routers), and a worker is
 * assigned to whichever router reaches it in the fewest hops -- but ONLY
 * if that router is the unique minimum (a tie, or no router reaching it at
 * all, leaves the node ungrouped rather than guessing).
 *
 * This is a deliberate simplification of true dominance analysis (which
 * would need Lengauer-Tarjan on a CFG-like reduction) -- proportionate for
 * a visual aid that's off by default and never silently changes the
 * underlying graph data, only how it's drawn.
 */
export function computeRouterClusters(
  nodeIds: string[],
  edges: GraphEdgeData[],
  isRouter: Map<string, boolean>
): Map<string, string> {
  const adjacency = new Map<string, string[]>();
  for (const e of edges) {
    if (!adjacency.has(e.src)) adjacency.set(e.src, []);
    adjacency.get(e.src)!.push(e.tgt);
  }

  const routers = nodeIds.filter((id) => isRouter.get(id));
  const bestDistance = new Map<string, number>();
  const bestRouter = new Map<string, string>();
  const tied = new Set<string>();

  for (const router of routers) {
    const dist = new Map<string, number>([[router, 0]]);
    const queue: string[] = [router];
    let head = 0;
    while (head < queue.length) {
      const cur = queue[head++];
      const d = dist.get(cur)!;
      if (cur !== router && isRouter.get(cur)) continue; // don't cross into another router's territory
      for (const next of adjacency.get(cur) ?? []) {
        if (next === "__start__" || next === "__end__") continue;
        if (!dist.has(next)) {
          dist.set(next, d + 1);
          queue.push(next);
        }
      }
    }

    for (const [node, d] of dist) {
      if (node === router) continue;
      const currentBest = bestDistance.get(node);
      if (currentBest === undefined || d < currentBest) {
        bestDistance.set(node, d);
        bestRouter.set(node, router);
        tied.delete(node);
      } else if (d === currentBest && bestRouter.get(node) !== router) {
        tied.add(node);
      }
    }
  }

  const clusterOf = new Map<string, string>();
  for (const [node, router] of bestRouter) {
    if (isRouter.get(node)) continue; // routers own clusters, they aren't owned by one
    if (tied.has(node)) continue; // ambiguous ownership -> leave ungrouped
    clusterOf.set(node, router);
  }
  return clusterOf;
}
