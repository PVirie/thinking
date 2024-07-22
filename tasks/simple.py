import os
import logging
import contextlib
import random
import json
from typing import List
from pydantic import BaseModel

from humn import *
from src.utilities import *
from src.jax import algebric as alg
from src.jax import cortex, hippocampus
from src.core import table


class Context(BaseModel):
    layers: List[(cortex.Model, hippocampus.Model)]
    states: alg.State_Sequence
    goals: List[int]
    graph: np.ndarray
    random_seed: int = 0


    @staticmethod
    def load(path):
        try:
            with open(os.path.join(path, "metadata.json"), "r") as f:
                metadata = json.load(f)
                layers = []
                for i in range(metadata["num_layers"]):
                    layer_path = os.path.join(path, "layers", f"layer_{i}")
                    cortex_path = os.path.join(layer_path, "cortex")
                    hippocampus_path = os.path.join(layer_path, "hippocampus")
                    layers.append((cortex.Model.load(cortex_path), hippocampus.Model.load(hippocampus_path)))
                states = alg.State_Sequence([alg.State.load(os.path.join(path, "states", f"state_{i}")) for i in range(metadata["num_states"])])
                graph = np.load(os.path.join(path, "graph.npy"))
                context = Context(layers=layers, states=states, goals=metadata["goals"], graph=graph, random_seed=metadata["random_seed"])
            return context
        except Exception as e:
            logging.error(e)
        return None
                                 

    def save(self, path):
        for i, layer in enumerate(self.layers):
            layer_path = os.path.join(path, "layers", f"layer_{i}")
            cortex_path = os.path.join(layer_path, "cortex")
            hippocampus_path = os.path.join(layer_path, "hippocampus")
            layer[0].save(cortex_path)
            layer[1].save(hippocampus_path)
        for i, state in enumerate(self.states):
            state_path = os.path.join(path, "states", f"state_{i}")
            state.save(state_path)
        graph_path = os.path.join(path, "graph.npy")
        np.save(graph_path, self.graph)
        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump({
                "num_layers": len(self.layers),
                "num_states": len(self.states),
                "goals": self.goals,
                "random_seed": self.random_seed
            }, f, indent=4)
        return True


    @staticmethod
    def setup():
        random_seed = random.randint(0, 1000)
        random.seed(random_seed)

        graph_shape = 16
        one_hot = generate_onehot_representation(np.arange(graph_shape), graph_shape)
        states = alg.State_Sequence(one_hot)

        graph = random_graph(graph_shape, 0.4)

        layers = [Layer(cortex.Model(table.Model(graph_shape)), hippocampus.Model(graph_shape, graph_shape)), Layer(cortex.Model(table.Model(graph_shape)), hippocampus.Model(graph_shape, graph_shape))]
        model = HUMN(layers)

        explore_steps = 10000
        logging.info("Training a cognitive map:")
        stamp = time.time()
        for i in range(explore_steps):
            path = random_walk(graph, random.randint(0, graph.shape[0] - 1), graph.shape[0] - 1)
            model.observe(states.generate_subsequence(alg.Index_Sequence(path)))
            if i % 100 == 0:
                logging.info(f"Training progress: {(i * 100 / explore_steps):.2f}", end="\r", flush=True)
        logging.info(f"\nFinish learning in {time.time() - stamp}s")

        return Context(model=model, states=states, goals=np.arange(graph_shape), graph=graph, random_seed=random_seed)



@contextlib.contextmanager
def experiment_session(path):
    cortex = Context.load(path)
    if cortex is None:
        cortex = Context.setup()
        context.save(path)
    yield context


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    artifact_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "artifacts")
    experiment_path = os.path.join(artifact_path, "experiments", "simple")

    max_steps = 40
    with experiment_session(experiment_path) as context:
        model = HUMN(context.layers)

        def exp_loop(preference = None):
            total_length = 0
            stamp = time.time()
            for t_i in context.goals:
                s = context.states[0]
                t = context.states[t_i]
                ps = []
                model.refresh()
                for i in range(max_steps):
                    p = model.think(s, t)
                    p_i = context.states[p]
                    ps.append(p_i)
                    total_length = total_length + 1
                    if p == t:
                        break
                    # enhance result
                    s = context.states[p_i]
                if i == max_steps - 1:
                    logging.warning("fail to find path in time.", t)
                logging.info(ps)
            return total_length, time.time() - stamp

        logging.info("-----------cognitive planner-----------")
        total_length, elapsed_seconds = exp_loop()
        logging.info("cognitive planner:", elapsed_seconds, " average length:", total_length / len(context.goals))

        logging.info("-----------optimal planner-----------")
        total_length = 0
        stamp = time.time()
        for t in context.goals:
            try:
                ps = shortest_path(context.graph, 0, t)
                ps = list(reversed(ps))
                total_length = total_length + len(ps)
            except RecursionError:
                logging.info("fail to find path in time.", t)
            finally:
                logging.info(ps)
        logging.info("optimal planner:", time.time() - stamp, " average length:", total_length / len(context.goals))

