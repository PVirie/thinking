import os
import logging
import contextlib
import random
import json
from typing import List, Any
from pydantic import BaseModel

from humn import *
from src.utilities import *
from src.jax import algebric as alg
from src.jax import cortex, hippocampus, abstraction
import src.core as core
from src.core import table, linear, linear_stat
import jax


class Context(BaseModel):
    layers: List[Any]
    states: Any
    goals: Any
    graph: Any
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
                    layer = Layer(cortex.Model.load(cortex_path), hippocampus.Model.load(hippocampus_path))
                    if os.path.exists(os.path.join(layer_path, "abstraction")):
                        abstraction_path = os.path.join(layer_path, "abstraction")
                        layer.abstraction_model = cortex.Model.load(abstraction_path)
                    layers.append(layer)
                states = alg.State_Sequence.load(os.path.join(path, "states"))
                goals = np.load(os.path.join(path, "goals.npy"))
                graph = np.load(os.path.join(path, "graph.npy"))
                context = Context(layers=layers, states=states, goals=goals, graph=graph, random_seed=metadata["random_seed"])
            return context
        except Exception as e:
            logging.error(e)
        return None
                                 

    @staticmethod
    def save(self, path):
        os.makedirs(path, exist_ok=True)
        for i, layer in enumerate(self.layers):
            layer_path = os.path.join(path, "layers", f"layer_{i}")
            cortex_path = os.path.join(layer_path, "cortex")
            hippocampus_path = os.path.join(layer_path, "hippocampus")
            cortex.Model.save(layer.cortex_model, cortex_path)
            hippocampus.Model.save(layer.hippocampus_model, hippocampus_path)
            if layer.abstraction_model is not None:
                abstraction_path = os.path.join(layer_path, "abstraction")
                abstraction.Model.save(layer.abstraction_model, abstraction_path)
        alg.State_Sequence.save(self.states, os.path.join(path, "states"))
        goal_path = os.path.join(path, "goals.npy")
        np.save(goal_path, self.goals)
        graph_path = os.path.join(path, "graph.npy")
        np.save(graph_path, self.graph)
        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump({
                "num_layers": len(self.layers),
                "num_states": len(self.states),
                "random_seed": self.random_seed
            }, f, indent=4)
        return True


    @staticmethod
    def setup(setup_path):
        random_seed = random.randint(0, 1000)
        random.seed(random_seed)

        graph_shape = 16
        one_hot = generate_onehot_representation(np.arange(graph_shape), graph_shape)
        states = alg.State_Sequence(one_hot)

        graph = random_graph(graph_shape, 0.4)

        layers = [
            Layer(cortex.Model(linear.Model(graph_shape, graph_shape)), hippocampus.Model(graph_shape, graph_shape)), 
            Layer(cortex.Model(linear.Model(graph_shape, graph_shape)), hippocampus.Model(graph_shape, graph_shape)), 
            Layer(cortex.Model(linear.Model(graph_shape, graph_shape)), hippocampus.Model(graph_shape, graph_shape)), 
            Layer(cortex.Model(linear.Model(graph_shape, graph_shape)), hippocampus.Model(graph_shape, graph_shape))
        ]

        # layers = []
        # for i in range(3):
        #     linear_core = linear.Model(graph_shape, graph_shape)
        #     c = cortex.Model(linear_core)
        #     h = hippocampus.Model(graph_shape, graph_shape)
        #     a = abstraction.Model(linear_stat.Model(linear_core))
        #     layers.append(Layer(c, h, a))

        model = HUMN(layers)

        explore_steps = 10000
        print_steps = max(1, explore_steps // 100)
        logging.info("Training a cognitive map:")
        stamp = time.time()
        for i in range(explore_steps):
            path = random_walk(graph, random.randint(0, graph.shape[0] - 1), graph.shape[0] - 1)
            model.observe(states.generate_subsequence(alg.Pointer_Sequence(path)))
            if i % print_steps == 0 and i > 0:
                # print at every 1 % progress
                # compute time to finish in seconds
                logging.info(f"Training progress: {(i * 100 / explore_steps):.2f}, time to finish: {((time.time() - stamp) * (explore_steps - i) / i):.2f}s")
        logging.info(f"\nTotal learning time {time.time() - stamp}s")

        return Context(layers=layers, states=states, goals=np.arange(graph_shape), graph=graph, random_seed=random_seed)



@contextlib.contextmanager
def experiment_session(path, force_clear=False):
    core.initialize(os.path.join(path, "core"))
    if force_clear:
        empty_directory(path)
    context = Context.load(path)
    if context is None:
        context = Context.setup(path)
        Context.save(context, path)
    yield context


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    experiment_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "experiments", "simple")

    max_steps = 40
    with experiment_session(experiment_path) as context:
        model = HUMN(context.layers)

        def exp_loop(preference = None):
            total_length = 0
            stamp = time.time()
            for t_i in context.goals:
                s = context.states[0]
                t = context.states[t_i]
                ps = [0]
                model.refresh()
                for i in range(max_steps):
                    if s == t:
                        break
                    a = model.think(s, t - s)
                    p = s + a
                    p_i = context.states[p]
                    ps.append(p_i)
                    # enhance result
                    s = context.states[p_i]
                if i == max_steps - 1:
                    logging.warning("fail to find path in time.")
                    total_length = total_length + max_steps
                else:
                    total_length = total_length + len(ps)
                ps = list(map(int, ps))
                logging.info(f"s: {0} t: {t_i} {ps}")
            return total_length, time.time() - stamp

        logging.info("-----------cognitive planner-----------")
        total_length, elapsed_seconds = exp_loop()
        logging.info(f"cognitive planner: {elapsed_seconds:.2f}s, average length: {total_length / len(context.goals):.2f}")

        logging.info("-----------random planner-----------")
        total_length = 0
        stamp = time.time()
        for t in context.goals:
            ps = random_walk(context.graph, 0, max_steps)
            for i, p in enumerate(ps):
                if p == t:
                    break
            if p != t:
                logging.warning("fail to find path in time.")
                total_length = total_length + max_steps
            else:
                total_length = total_length + len(ps[:i+1])
            # to int list
            ps = list(map(int, ps))
            logging.info(f"s: {0} t: {t} {ps[:i+1]}")
        logging.info(f"random planner: {time.time() - stamp:.2f}s, average length: {total_length / len(context.goals):.2f}")

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
                ps = list(map(int, ps))
                logging.info(f"s: {0} t: {t} {ps}")
        logging.info(f"optimal planner: {time.time() - stamp:.2f}s, average length: {total_length / len(context.goals):.2f}")
