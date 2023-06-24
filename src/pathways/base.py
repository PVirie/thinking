import sys
import os
from typing import List

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, '..'))
from metric import Node

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

    async def enhance(self, input: Node):
        pass

    async def sample(self, x: Node, forward=True):
        pass
    