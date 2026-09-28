"""Minimum-spanning-tree regularization for deformable registration.

Ports the belief-propagation regularizer of deedsBCV (``primsMST.h`` +
``regularisation.h``): the discrete per-control-point data cost is regularized
by exact message passing on an image-driven minimum spanning tree of the
control grid. Because a tree is loop-free, one leaves→root sweep of
min-convolutions plus a root→leaves argmin back-substitution yields the globally
optimal labelling for the tree-approximated MRF.

Label convention (our own, not the reference's Fortran layout): a control
point's cost cube is ``(L, L, L)`` with ``L = 2*hw + 1`` and axes ``(dz, dy, dx)``;
label index ``a`` on an axis means displacement ``(a - hw) * quant`` voxels.
Everything here runs on the host (numpy) — it is graph-serial, not the FLOP
bottleneck (the data cost is).
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order, minimum_spanning_tree


# --------------------------------------------------------------------------- #
# Distance transform (separable squared-L2 min-convolution) — messageDT
# --------------------------------------------------------------------------- #

def _dt_1d(cost: np.ndarray, offset: np.ndarray, axis: int):
    """One separable pass: ``msg[...,i,...] = min_b cost[...,b,...] + (i-b+off)^2``.

    *cost* is ``(N, L, L, L)``; *offset* is ``(N,)`` (per-node). Returns the
    min-convolved messages and the per-axis argmin source index, same shape.
    Vectorized across nodes and lines by looping over the L source labels.
    """
    n = cost.shape[0]
    L = cost.shape[axis]
    c = np.moveaxis(cost, axis, 1).reshape(n, L, -1)  # (N, L, R)
    R = c.shape[2]
    idx = np.arange(L, dtype=np.float32).reshape(1, L)
    off = offset.astype(np.float32).reshape(n, 1)

    msg = np.full((n, L, R), np.inf, dtype=np.float32)
    arg = np.zeros((n, L, R), dtype=np.int16)
    for b in range(L):
        pen = (idx - b + off) ** 2                    # (N, L)
        cand = c[:, b:b + 1, :] + pen[:, :, None]     # (N, L, R)
        better = cand < msg
        msg = np.where(better, cand, msg)
        arg = np.where(better, np.int16(b), arg)

    others = tuple(s for i, s in enumerate(cost.shape) if i not in (0, axis))
    msg = np.moveaxis(msg.reshape((n, L) + others), 1, axis)
    arg = np.moveaxis(arg.reshape((n, L) + others), 1, axis)
    return msg, arg


def message_dt_batch(cost: np.ndarray, offsets: np.ndarray):
    """Batched 3-D messageDT over a set of control points.

    *cost* ``(N, L, L, L)`` axes ``(dz, dy, dx)``; *offsets* ``(N, 3)`` are the
    parent-relative label offsets ``(u0_parent - u0_child)/quant`` per axis.
    Returns ``(messages (N,L,L,L), src_flat (N,L,L,L))`` where ``src_flat`` maps
    each *output* (parent) label to the child's optimal *source* label (flat
    C-order index into the ``(L,L,L)`` cube), for root→leaves back-substitution.
    """
    n, L = cost.shape[0], cost.shape[1]
    m0, a0 = _dt_1d(cost, offsets[:, 0], axis=1)   # reduce dz
    m1, a1 = _dt_1d(m0, offsets[:, 1], axis=2)     # reduce dy
    m2, a2 = _dt_1d(m1, offsets[:, 2], axis=3)     # reduce dx

    # Compose per-axis argmins into the full source label for each output label.
    s2 = a2                                          # source dx index
    s1 = np.take_along_axis(a1, s2, axis=3)          # source dy index
    ng = np.arange(n).reshape(n, 1, 1, 1)
    o0 = np.arange(L).reshape(1, L, 1, 1)
    s0 = a0[ng, o0, s1, s2]                           # source dz index
    src_flat = (s0.astype(np.int32) * L + s1) * L + s2
    return m2, src_flat


# --------------------------------------------------------------------------- #
# Image-driven minimum spanning tree — primsGraph
# --------------------------------------------------------------------------- #

def _block_mean_sad(vol: np.ndarray, step: int, axis: int, grid_shape):
    """Mean |ΔI| between each control block and its +1 neighbour along *axis*.

    Returns a ``grid_shape`` array; the last slab along *axis* (no forward
    neighbour) is left at 0 and not used as an edge.
    """
    gz, gy, gx = grid_shape
    cz, cy, cx = gz * step, gy * step, gx * step
    v = vol[:cz, :cy, :cx].astype(np.float32)

    # |I(p) - I(p + step·e_axis)| over the valid region, then block-mean.
    sl_lo = [slice(None)] * 3
    sl_hi = [slice(None)] * 3
    sl_lo[axis] = slice(0, v.shape[axis] - step)
    sl_hi[axis] = slice(step, v.shape[axis])
    diff = np.zeros_like(v)
    valid = [slice(None)] * 3
    valid[axis] = slice(0, v.shape[axis] - step)
    diff[tuple(valid)] = np.abs(v[tuple(sl_lo)] - v[tuple(sl_hi)])

    sad = diff.reshape(gz, step, gy, step, gx, step).mean(axis=(1, 3, 5))
    return sad


def prims_graph(grid_vol: np.ndarray, step: int, grid_shape):
    """Image-driven MST over the control grid (port of ``primsGraph``).

    Edge weight between neighbouring control points is derived from the mean
    absolute intensity difference (SAD) of their ``step^3`` blocks:
    ``similarity = exp(-SAD / (2·std))`` — similar blocks give strong (cheap)
    edges so the tree avoids crossing genuine boundaries.

    Returns ``(order, parents, edgemst)``:
      * ``order`` — BFS order from the centre root (root first),
      * ``parents`` — parent node id per node (root's is itself),
      * ``edgemst`` — per-child ``similarity`` weight to its parent (root: 0).
    """
    gz, gy, gx = grid_shape
    n = gz * gy * gx
    vol = np.asarray(grid_vol, dtype=np.float32)
    std = float(vol.std()) or 1.0

    def cp_id(z, y, x):
        return (z * gy + y) * gx + x

    rows, cols, wts, sims = [], [], [], []
    # Three forward directions cover all 6-connected edges once.
    for axis, (dz, dy, dx) in enumerate([(1, 0, 0), (0, 1, 0), (0, 0, 1)]):
        sad = _block_mean_sad(vol, step, axis, grid_shape)
        sim = np.exp(-sad / (2.0 * std))              # (gz,gy,gx) in (0,1]
        zz, yy, xx = np.meshgrid(
            np.arange(gz), np.arange(gy), np.arange(gx), indexing="ij"
        )
        # valid where a +1 neighbour exists along this axis
        keep = {0: zz < gz - 1, 1: yy < gy - 1, 2: xx < gx - 1}[axis]
        a = cp_id(zz, yy, xx)[keep]
        b = cp_id(zz + dz, yy + dy, xx + dx)[keep]
        s = sim[keep]
        rows.append(a); cols.append(b); sims.append(s)
        # Positive graph weight, order-preserving vs the reference's -sim, with a
        # constant + eps (MST-invariant) so no strictly-zero weights are dropped.
        wts.append((1.0 - s) + 1e-6)

    rows = np.concatenate(rows); cols = np.concatenate(cols)
    wts = np.concatenate(wts); sims = np.concatenate(sims)

    graph = csr_matrix((wts, (rows, cols)), shape=(n, n))
    mst = minimum_spanning_tree(graph)               # directed (upper) tree
    mst_sym = mst + mst.T                             # undirected for traversal

    root = cp_id(gz // 2, gy // 2, gx // 2)
    order, preds = breadth_first_order(
        mst_sym, root, directed=False, return_predecessors=True
    )
    parents = preds.copy()
    parents[root] = root

    # edgemst[child] = similarity on the (child, parent) edge.
    sim_lookup = {}
    for a, b, s in zip(rows, cols, sims):
        sim_lookup[(int(a), int(b))] = float(s)
        sim_lookup[(int(b), int(a))] = float(s)
    edgemst = np.zeros(n, dtype=np.float32)
    for node in order[1:]:
        edgemst[node] = sim_lookup.get((int(node), int(parents[node])), 0.0)

    return order.astype(np.int64), parents.astype(np.int64), edgemst


# --------------------------------------------------------------------------- #
# Tree belief propagation — regularisationCL
# --------------------------------------------------------------------------- #

def regularise(costall, u0, order, parents, edgemst, hw, quant, grid_shape):
    """Belief-propagation regularization over the MST (port of ``regularisationCL``).

    Parameters
    ----------
    costall : (N, L, L, L) float32
        Per-control-point data cost; mutated in place.
    u0 : (3, gz, gy, gx)
        Carried displacement field this level refines (absolute voxels).
    order, parents, edgemst : from :func:`prims_graph`.
    hw, quant : label half-width and step of this level.

    Returns the updated field ``(3, gz, gy, gx)``.
    """
    costall = np.ascontiguousarray(costall, dtype=np.float32)
    n, L = costall.shape[0], costall.shape[1]
    u0_flat = np.asarray(u0, dtype=np.float32).reshape(3, -1)

    root = int(order[0])
    depth = np.zeros(n, dtype=np.int64)
    for node in order[1:]:
        depth[node] = depth[parents[node]] + 1
    maxlev = int(depth.max()) + 1
    levels = [np.where(depth == lev)[0] for lev in range(maxlev)]

    allsrc = np.zeros((n, L * L * L), dtype=np.int32)

    # ---- leaves -> root ------------------------------------------------
    for lev in range(maxlev - 1, 0, -1):
        nodes = levels[lev]
        if nodes.size == 0:
            continue
        par = parents[nodes]
        costall[nodes] *= edgemst[nodes].reshape(-1, 1, 1, 1)
        offs = ((u0_flat[:, par] - u0_flat[:, nodes]) / float(quant)).T  # (K,3)
        msg, src = message_dt_batch(costall[nodes], offs)
        costall[nodes] = msg
        allsrc[nodes] = src.reshape(nodes.size, -1)
        contrib = msg - msg.reshape(nodes.size, -1).min(axis=1).reshape(-1, 1, 1, 1)
        np.add.at(costall, par, contrib)

    # ---- root argmin ----------------------------------------------------
    selected = np.zeros(n, dtype=np.int64)
    selected[root] = int(costall[root].reshape(-1).argmin())

    # ---- root -> leaves back-substitution ------------------------------
    for lev in range(1, maxlev):
        nodes = levels[lev]
        if nodes.size == 0:
            continue
        par = parents[nodes]
        selected[nodes] = allsrc[nodes, selected[par]]

    # ---- assemble field -------------------------------------------------
    a0, a1, a2 = np.unravel_index(selected, (L, L, L))
    disp = np.stack([a0, a1, a2]).astype(np.float32) - hw
    disp *= float(quant)
    field = (disp + u0_flat).reshape(3, *grid_shape)
    return field.astype(np.float32)
