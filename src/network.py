import os
from typing import List
from .metric import Node, No_Printer
from loguru import logger

def compute_pivot_indices(entropies):
    all_pvs = []
    for j in range(1, len(entropies) - 1):
        if entropies[j - 1] < entropies[j] and entropies[j] >= entropies[j + 1]:
            all_pvs.append(j)
    all_pvs.append(len(entropies) - 1)
    return all_pvs

class Layer:
    def __init__(self, name, heuristics, hippocampus, proxy):
        self.name = name
        self.heuristics = heuristics
        self.hippocampus = hippocampus
        self.proxy = proxy
        self.next = None

    def assign_next(self, next_layer):
        self.next = next_layer

    def save(self, weight_path, index=0):
        layer_path = os.path.join(weight_path, f"layer_{index}")
        os.makedirs(layer_path, exist_ok=True)
        self.heuristics.save(layer_path)
        self.hippocampus.save(layer_path)
        self.proxy.save(layer_path)
        
        if self.next is not None:
            self.next.save(weight_path, index + 1)

    def load(self, weight_path, index=0):
        layer_path = os.path.join(weight_path, f"layer_{index}")
        self.heuristics.load(layer_path)
        self.hippocampus.load(layer_path)
        self.proxy.load(layer_path)

        if self.next is not None:
            self.next.load(weight_path, index + 1)

    async def incrementally_learn(self, path: List[Node]):
        if len(path) < 2:
            return

        await self.hippocampus.incrementally_learn(path)
        await self.proxy.incrementally_learn(path)

        entropies = await self.hippocampus.compute_entropy(path)
        pivots_indices = compute_pivot_indices(entropies)
        pivots = [path[i] for i in pivots_indices]

        # await self.heuristics.incrementally_learn(path, pivots_indices)
        hippocampus_distances = await self.hippocampus.distance(path, [path[i] for i in pivots_indices])
        await self.heuristics.incrementally_learn_2(path, pivots_indices, hippocampus_distances)

        if self.next is not None and len(pivots) > 1:
            await self.next.incrementally_learn(pivots)

    async def to_next(self, c: Node, forward=True):
        # not just enhance, but select the closest with highest entropy

        entropy = (await self.hippocampus.compute_entropy([c]))[0]
        for i in range(1000):
            next_c = await self.hippocampus.sample(c, forward)
            next_entropy = (await self.hippocampus.compute_entropy([c]))[0]

            if entropy >= next_entropy:
                return c
            c = next_c
            entropy = next_entropy

        raise ValueError('Cannot find a top node in time.')

    async def from_next(self, c: Node):
        return c

    async def pincer_inference(self, s: Node, t: Node, pathway_bias=0):
        # pathway_bias < 0 : use hippocampus
        # pathway_bias > 0 : use cortex

        candidates, props = await self.proxy.get_candidates(s)

        cortex_rep, cortex_prop = await self.heuristics.consolidate(s, candidates, props, t)
        hippocampus_rep, hippocampus_prop = await self.hippocampus.infer(s, t)

        sup_info = [(await self.hippocampus.enhance(cortex_rep), cortex_prop), (await self.hippocampus.enhance(hippocampus_rep), hippocampus_prop)]

        if pathway_bias < 0:
            return hippocampus_rep, sup_info
        if pathway_bias > 0:
            return cortex_rep, sup_info

        return hippocampus_rep if hippocampus_prop > cortex_prop else cortex_rep, sup_info

    async def find_path(self, c: Node, t: Node, hard_limit=20, pathway_bias=0, printer=No_Printer()):
        # pathway_bias < 0 : use hippocampus
        # pathway_bias > 0 : use cortex

        count_steps = 0

        await printer.print({
            "layer": self.name,
            "s": c,
            "t": t
        })
            
        if self.next is not None:
            goals = self.next.find_path(await self.to_next(c, True), await self.to_next(t, False), hard_limit=hard_limit, pathway_bias=pathway_bias, printer=printer)

        yield c

        while True:
            if await c.is_same_node(t):
                # we have reached the target
                await printer.print("target reached!")
                break

            if self.next is not None:
                try:
                    data = await anext(goals)
                    g = await self.from_next(data)
                except StopIteration:
                    g = t
            else:
                g = t
            
            await self.proxy.reset()

            while True:
                if await c.is_same_node(t):
                    # we have reached the target
                    await printer.print("target reached!")
                    break

                if await c.is_same_node(g):
                    # we have reached the goal
                    await printer.print("goal reached.")
                    break

                count_steps = count_steps + 1
                if count_steps >= hard_limit:
                    await printer.print({
                        "layer": self.name,
                        "success": False,
                        "reason": "hard limit reached."
                    })
                    raise RecursionError

                c, supplementary = await self.pincer_inference(c, g, pathway_bias)
                c = await self.hippocampus.enhance(c)  # enhance signal preventing signal degradation

                # this may contradict with ONLINE LEARNING, luckily we don't do it now.
                await self.proxy.update_visit(c)

                await printer.print({
                    "layer": self.name,
                    "goal": g,
                    "selected": c,
                    "choices": supplementary
                })
                yield c


    async def next_step(self, c: Node, t: Node, pathway_bias=0):
        # pathway_bias < 0 : use hippocampus
        # pathway_bias > 0 : use cortex

        if await c.is_same_node(t):
            return t

        next_g = await self.next.next_step(await self.to_next(c), await self.to_next(t))
        g = await self.from_next(next_g)

        if await c.is_same_node(g):
            return g

        c, supplementary = await self.pincer_inference(c, g, pathway_bias)
        return c
