from src import network
import uuid
import asyncio


class Heuristics_service:
    def __init__(self):
        pass

    async def incrementally_learn(self, path):
        pass

    async def compute_entropy(self, path):
        pass

    async def get_distinct_next_candidate(self, c, forward=True):
        pass

    async def consolidate(self, s, candidates, props, t):
        pass

    async def infer(self, s, t):
        pass

    async def enhance(self, s):
        pass

class Hippocampus_service:
    def __init__(self):
        pass

    async def incrementally_learn(self, path):
        pass

    async def compute_entropy(self, path):
        pass

    async def get_distinct_next_candidate(self, c, forward=True):
        pass

    async def infer(self, s, t):
        pass

    async def enhance(self, s):
        pass


class Embedding_service:
    def __init__(self):
        pass

    async def incrementally_learn(self, path):
        pass


class Session:

    def __init__(self, config):
        self.id = uuid.uuid4()

        layers = []
        for i in range(config["num_layers"]):
            layers.append(network.Layer(Heuristics_service(), Hippocampus_service()))

        for i in range(len(layers) - 1):
            layers[i].assign_next(layers[i + 1])

        self.cognitive_map = layers[0]


    async def incrementally_learn(self, path):
        await self.cognitive_map.incrementally_learn(path)


    async def find_path(self, c, t, hard_limit=20, pathway_bias=0):
        return await self.cognitive_map.find_path(c, t, hard_limit, pathway_bias)


    async def next_step(self, c, t, pathway_bias=0):
        return await self.cognitive_map.next_step(c, t, pathway_bias)