from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.fx as fx

from graph.ir import GraphIR


_OP_MAP = {
    nn.Conv2d: "Conv2D",
    nn.ReLU: "ReLU",
    nn.MaxPool2d: "MaxPool",
    nn.Linear: "Dense",
}


def _torch_fx_target_to_op(modules: Dict[str, nn.Module], node: fx.Node) -> Tuple[str, Dict[str, object]]:
    """
    Map an FX node to a compact op name + lightweight attrs.
    """
    if node.op == "placeholder":
        return "Input", {"dtype": str(node.meta.get("tensor_meta", {}).dtype) if node.meta else None}

    if node.op == "call_module":
        m = modules[str(node.target)]
        for t, name in _OP_MAP.items():
            if isinstance(m, t):
                attrs: Dict[str, object] = {}
                if isinstance(m, nn.Conv2d):
                    attrs = {
                        "in_channels": m.in_channels,
                        "out_channels": m.out_channels,
                        "kernel_size": m.kernel_size,
                        "stride": m.stride,
                        "padding": m.padding,
                    }
                elif isinstance(m, nn.MaxPool2d):
                    attrs = {"kernel_size": m.kernel_size, "stride": m.stride, "padding": m.padding}
                elif isinstance(m, nn.Linear):
                    attrs = {"in_features": m.in_features, "out_features": m.out_features}
                return name, attrs
        return type(m).__name__, {"target": str(node.target)}

    if node.op == "call_function":
        # flatten shows up as torch.flatten in the sample model; keep it but treat as layout op.
        if node.target == torch.flatten:
            return "Flatten", {}
        return getattr(node.target, "__name__", str(node.target)), {}

    if node.op == "output":
        return "Output", {}

    return node.op, {"target": str(node.target)}


def extract_graph(model: nn.Module, example_input: torch.Tensor) -> GraphIR:
    """
    Extract a lightweight graph from a PyTorch model using torch.fx.

    Output is intentionally minimal: nodes, edges (via inputs), and small attrs.
    """
    gm = fx.symbolic_trace(model)

    # Propagate shapes/dtypes where possible (best-effort).
    try:
        fx.passes.shape_prop.ShapeProp(gm).propagate(example_input)
    except Exception:
        pass

    modules = dict(gm.named_modules())
    gir = GraphIR()

    fx_to_ir: Dict[fx.Node, int] = {}
    for n in gm.graph.nodes:
        op_name, attrs = _torch_fx_target_to_op(modules, n)

        # Convert arguments to IR input ids (only handle tensor-producing node deps).
        inputs: List[int] = []
        for a in n.all_input_nodes:
            if a in fx_to_ir:
                inputs.append(fx_to_ir[a])

        ir_id = gir.add_node(name=n.name, op=op_name, inputs=inputs, attrs=attrs)
        fx_to_ir[n] = ir_id

    return gir

