from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class Node:
    id: int
    name: str
    op: str  # e.g. "Input", "Conv2D", "ReLU", "MaxPool", "Dense"
    inputs: List[int]
    attrs: Dict[str, object]


class GraphIR:
    def __init__(self) -> None:
        self.nodes: List[Node] = []
        self._name_to_id: Dict[str, int] = {}

    def add_node(
        self,
        *,
        name: str,
        op: str,
        inputs: Optional[List[int]] = None,
        attrs: Optional[Dict[str, object]] = None,
    ) -> int:
        node_id = len(self.nodes)
        n = Node(
            id=node_id,
            name=name,
            op=op,
            inputs=list(inputs or []),
            attrs=dict(attrs or {}),
        )
        self.nodes.append(n)
        self._name_to_id[name] = node_id
        return node_id

    def node(self, node_id: int) -> Node:
        return self.nodes[node_id]

    def pretty(self) -> str:
        """
        Human-friendly view, emphasizing the main dataflow chain.
        """
        if not self.nodes:
            return "<empty graph>"

        # Heuristic: choose the first node with op=="Input" as the start, then
        # follow single-user edges where possible. Good enough for the sample model.
        start = next((n.id for n in self.nodes if n.op == "Input"), 0)
        users: Dict[int, List[int]] = {n.id: [] for n in self.nodes}
        for n in self.nodes:
            for inp in n.inputs:
                users[inp].append(n.id)

        chain = [start]
        cur = start
        visited = {start}
        while True:
            nxts = [u for u in users.get(cur, []) if u not in visited]
            if len(nxts) != 1:
                break
            cur = nxts[0]
            visited.add(cur)
            chain.append(cur)

        lines = []
        for i, nid in enumerate(chain):
            n = self.node(nid)
            lines.append(n.op)
            if i != len(chain) - 1:
                # ASCII-only so this prints cleanly on any console encoding.
                lines.append("->")
        return "\n".join(lines)

