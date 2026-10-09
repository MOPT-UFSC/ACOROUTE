"""
grid.py
3D lattice of candidate nodes for tube-path design (procedural style).

A path is a sequence of lattice nodes from A to B. Nodes are connected to their
neighbours within one cell along each axis (26-connectivity by default, so the tube
may run along faces, edges or body diagonals of the lattice cells, matching the
neighbourhood drawing in the original talk). Arbitrary (non-box) admissible regions
are supported through an optional `mask(ix,iy,iz) -> bool`.
"""

import numpy as np
from itertools import product


def build_grid(nx, ny, nz, spacing, origin=(0.0, 0.0, 0.0), mask=None):
    """Build a regular lattice.

    Returns a dict:
        coords : (N,3) world coordinates of active nodes
        ijk    : (N,3) integer lattice indices of active nodes
        id_of  : dict (ix,iy,iz) -> node id
        nx,ny,nz,spacing : echoed parameters
    `mask` (optional) keeps only nodes where mask(ix,iy,iz) is True.
    """
    sx, sy, sz = (spacing, spacing, spacing) if np.isscalar(spacing) else spacing
    ox, oy, oz = origin
    coords, ijk, id_of = [], [], {}
    nid = 0
    for ix, iy, iz in product(range(nx), range(ny), range(nz)):
        if mask is not None and not mask(ix, iy, iz):
            continue
        coords.append((ox + ix * sx, oy + iy * sy, oz + iz * sz))
        ijk.append((ix, iy, iz))
        id_of[(ix, iy, iz)] = nid
        nid += 1
    return {
        "coords": np.array(coords, float),
        "ijk": np.array(ijk, int),
        "id_of": id_of,
        "nx": nx, "ny": ny, "nz": nz, "spacing": (sx, sy, sz),
    }


def build_neighbours(grid, connectivity=26):
    """Adjacency list: node id -> list of (neighbour id, segment length).

    connectivity in {6, 18, 26}.
    """
    if connectivity == 6:
        offs = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
    else:
        offs = [d for d in product((-1,0,1), repeat=3) if d != (0,0,0)]
        if connectivity == 18:
            offs = [d for d in offs if sum(abs(c) for c in d) <= 2]

    id_of = grid["id_of"]
    sx, sy, sz = grid["spacing"]
    adj = {}
    for (ix, iy, iz), nid in id_of.items():
        lst = []
        for dx, dy, dz in offs:
            key = (ix + dx, iy + dy, iz + dz)
            mid = id_of.get(key)
            if mid is not None:
                L = np.sqrt((dx*sx)**2 + (dy*sy)**2 + (dz*sz)**2)
                lst.append((mid, L))
        adj[nid] = lst
    return adj


def edge_key(i, j):
    """Undirected edge key (pheromone is stored per geometric edge)."""
    return (i, j) if i < j else (j, i)


def path_coords(grid, path_ids):
    """World coordinates of a path given as a list of node ids."""
    return grid["coords"][np.asarray(path_ids, int)]


def path_length(grid, path_ids):
    """Total Euclidean length of a node-id path."""
    c = path_coords(grid, path_ids)
    return float(np.sum(np.linalg.norm(np.diff(c, axis=0), axis=1)))


def distances_from(adj, source_id):
    """Shortest-path (geodesic) distance from `source_id` to every reachable node.

    One Dijkstra sweep over the weighted adjacency; returns a dict node_id -> distance.
    Used by the balanced detour budget in aco.build_one_ant: on a masked lattice the
    straight-line distance ignores detours forced by obstacles (baffles, keep-outs),
    whereas this graph distance is the exact minimum remaining polyline length, so
    mandatory detours are never charged against the discretionary slack.
    """
    import heapq
    dist = {source_id: 0.0}
    pq = [(0.0, source_id)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, float("inf")):
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist


def shortest_path(adj, start_id, target_id):
    """Minimum-length node path from start to target on a weighted adjacency.

    `adj` maps a node id to a list of (neighbour_id, edge_length). Returns the path as
    a list of node ids, or None if the target is unreachable. Used to warm-start the ACO:
    the shortest route reaches the target cheaply and biases the search toward short,
    direct routes, which a purely random construction struggles to find on a fine lattice.
    """
    import heapq
    dist = {start_id: 0.0}
    prev = {start_id: None}
    pq = [(0.0, start_id)]
    while pq:
        d, u = heapq.heappop(pq)
        if u == target_id:
            break
        if d > dist.get(u, float("inf")):
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    if target_id not in prev:
        return None
    path = []
    u = target_id
    while u is not None:
        path.append(u)
        u = prev[u]
    return path[::-1]


def _loop_erase(path):
    """Erase cycles from a node walk, keeping a simple path (first-occurrence cut)."""
    out, seen = [], {}
    for u in path:
        if u in seen:
            k = seen[u]
            for v in out[k + 1:]:
                seen.pop(v, None)
            out = out[:k + 1]
        else:
            seen[u] = len(out)
            out.append(u)
    return out


def diverse_seed_paths(adj, coords, start_id, target_id, k=3, n_via=8):
    """Up to `k` short, mutually distinct lattice routes from start to target, for
    warm-starting the ACO with several pheromone corridors instead of one.

    Seeding only the single shortest path biases the early search toward one corridor;
    on a domain with genuinely different route families (e.g. passing a gap near one
    wall or the other) that is an exploration bottleneck. This returns the shortest
    path plus alternatives forced through via-nodes chosen far from it: the nodes most
    distant from the shortest path are found, `n_via` of them are spread out by
    farthest-point sampling, and for each the concatenation shortest(start->via) +
    shortest(via->target) is loop-erased to a simple path. Duplicates are dropped and
    the `k` shortest distinct routes are returned (the plain shortest path always
    first, so warm-start reachability is never worse than with one corridor). Purely
    graph-based and deterministic: no FEM, a handful of Dijkstra runs."""
    import numpy as np
    base = shortest_path(adj, start_id, target_id)
    if base is None:
        return []
    routes = [base]
    if k <= 1:
        return routes
    base_xyz = coords[np.asarray(base)]
    ids = np.fromiter(adj.keys(), dtype=int)
    # distance of every node to the shortest path (to its nodes; enough to spread out)
    d2 = ((coords[ids][:, None, :] - base_xyz[None, :, :]) ** 2).sum(-1).min(1)
    order = ids[np.argsort(-d2)]
    cand = list(order[:max(4 * n_via, n_via)])
    # farthest-point sampling among the candidates, so the via-nodes are spread out
    vias = [cand[0]]
    while len(vias) < n_via and len(vias) < len(cand):
        dmin = None
        for c in cand:
            dd = min(((coords[c] - coords[v]) ** 2).sum() for v in vias)
            if dmin is None or dd > dmin[0]:
                dmin = (dd, c)
        if dmin is None or dmin[1] in vias:
            break
        vias.append(dmin[1])
    scored = []
    seen = {tuple(base)}
    for v in vias:
        p1 = shortest_path(adj, start_id, v)
        p2 = shortest_path(adj, v, target_id)
        if p1 is None or p2 is None:
            continue
        p = _loop_erase(p1 + p2[1:])
        if p[0] != start_id or p[-1] != target_id:
            continue
        key = tuple(p)
        if key in seen:
            continue
        seen.add(key)
        L = float(sum(np.linalg.norm(coords[p[i + 1]] - coords[p[i]])
                      for i in range(len(p) - 1)))
        scored.append((L, p))
    scored.sort(key=lambda t: t[0])
    routes.extend(p for _, p in scored[:k - 1])
    return routes

