"""
Operator fusion pass (prototype).

Initial target:
  Conv2D -> ReLU  ==>  Conv2DReLU (single fused op)

This reduces:
  - intermediate global memory writes/reads
  - kernel launch overhead
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Set

from graph.ir import GraphIR, Node


def fuse_conv_relu(gir: GraphIR) -> GraphIR:
    """
    Pattern-rewrite pass:

      Conv2D(x, w, ...) -> ReLU(.)   ==>   Conv2DReLU(x, w, ...)

    Preconditions for a safe local rewrite in this toy IR:
    - Conv2D has exactly one user, and that user is ReLU
    - ReLU has exactly one input (the Conv2D output)
    """

    if not gir.nodes:
        return gir

    users: Dict[int, List[int]] = {n.id: [] for n in gir.nodes}
    for n in gir.nodes:
        for inp in n.inputs:
            users[inp].append(n.id)

    skip: Set[int] = set()
    fused_of_conv: Dict[int, Node] = {}

    for conv in gir.nodes:
        if conv.op != "Conv2D":
            continue
        u = users.get(conv.id, [])
        if len(u) != 1:
            continue
        relu_id = u[0]
        relu = gir.node(relu_id)
        if relu.op != "ReLU":
            continue
        if relu.inputs != [conv.id]:
            continue

        fused = Node(
            id=-1,  # reassigned during rebuild
            name=f"{conv.name}_relu_fused",
            op="Conv2DReLU",
            inputs=list(conv.inputs),
            attrs={
                **dict(conv.attrs),
                "fused": True,
                "fused_ops": ("Conv2D", "ReLU"),
            },
        )
        fused_of_conv[conv.id] = fused
        skip.add(conv.id)
        skip.add(relu.id)

    if not fused_of_conv:
        return gir

    # Rebuild nodes in original order, replacing Conv2D with fused op and
    # dropping the ReLU.
    old_to_new: Dict[int, int] = {}
    new_nodes: List[Node] = []

    for old in gir.nodes:
        if old.id in skip:
            # keep only the fused replacement at the Conv2D position
            if old.id in fused_of_conv:
                new_id = len(new_nodes)
                n = replace(fused_of_conv[old.id], id=new_id)
                old_to_new[old.id] = new_id
                new_nodes.append(n)
            continue

        new_id = len(new_nodes)
        old_to_new[old.id] = new_id
        new_nodes.append(replace(old, id=new_id))

    # Now fix up inputs: any consumer of the removed ReLU should depend on the
    # fused node instead. Everything else remaps by old_to_new.
    # Build a map from removed ReLU id -> its Conv2D id (which becomes fused).
    relu_to_conv: Dict[int, int] = {}
    for conv_id, fused in fused_of_conv.items():
        # the relu was the only user of this conv
        relu_id = users[conv_id][0]
        relu_to_conv[relu_id] = conv_id

    fixed_nodes: List[Node] = []
    for n in new_nodes:
        remapped_inputs: List[int] = []
        for inp in n.inputs:
            if inp in relu_to_conv:
                inp = relu_to_conv[inp]
            remapped_inputs.append(old_to_new[inp])
        fixed_nodes.append(replace(n, inputs=remapped_inputs))

    out = GraphIR()
    out.nodes = fixed_nodes
    return out


