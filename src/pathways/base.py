from typing import List
from node import Node

class Pathway:
    """Base class for all pathways."""

    def __init__(self):
        pass

    async def incrementally_learn(self, path: List[Node], pivots):
        pass

    async def consolidate(self, start: Node, candidates: List[Node], props: List[float], target: Node):
        pass

    async def infer(self, start: Node, target: Node):
        pass

    async def compute_entropy(self, path: List[Node]):
        pass

        