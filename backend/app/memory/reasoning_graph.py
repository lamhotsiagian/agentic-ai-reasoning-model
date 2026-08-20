from pydantic import BaseModel, Field
from typing import List, Optional

class ReasoningNode(BaseModel):
    node_id: str
    node_type: str = Field(description="One of [goal, action, memory, observation]")
    description: str
    parent_id: Optional[str] = None

class ReasoningGraph(BaseModel):
    """
    Traces the structural reasoning chain of the agent during task execution.
    """
    nodes: List[ReasoningNode] = Field(default_factory=list)

    def add_step(self, node_id: str, node_type: str, description: str, parent_id: Optional[str] = None):
        node = ReasoningNode(
            node_id=node_id,
            node_type=node_type,
            description=description,
            parent_id=parent_id
        )
        self.nodes.append(node)

    def trace_path_to_root(self, leaf_id: str) -> List[ReasoningNode]:
        path = []
        current_id = leaf_id
        while current_id:
            node = next((n for n in self.nodes if n.node_id == current_id), None)
            if not node:
                break
            path.append(node)
            current_id = node.parent_id
        return path
