import os
import logging
import contextlib
import random
import pickle
from pydantic import BaseModel

from humn import *
from src.utilities import *
from src.jax import algebric as alg
from src.jax.pathway import cortex, hippocampus


class Context(BaseModel):
    model: HUMN
    states: list[alg.State]
    goals: list[int]
    graph: np.ndarray
    random_seed: int = 0


    def load(self, path):
        context.model.load(path)
        self.states = pickle.load(open(os.path.join(path, "states.pkl"), "rb"))
        self.goals = pickle.load(open(os.path.join(path, "goals.pkl"), "rb"))
        self.graph = pickle.load(open(os.path.join(path, "graph.pkl"), "rb"))
        self.random_seed = pickle.load(open(os.path.join(path, "random_seed.pkl"), "rb"))
                                 

    def save(self, path):
        context.model.save(path)
        pickle.dump(self.states, open(os.path.join(path, "states.pkl"), "wb"))
        pickle.dump(self.goals, open(os.path.join(path, "goals.pkl"), "wb"))
        pickle.dump(self.graph, open(os.path.join(path, "graph.pkl"), "wb"))
        pickle.dump(self.random_seed, open(os.path.join(path, "random_seed.pkl"), "wb"))


    def ready(self):
        return True
    
    
    def setup(self):
        self.random_seed = random.randint(0, 1000)
        random.seed(self.seed)

        graph_shape = 16
        one_hot = generate_onehot_representation(np.arange(graph_shape), graph_shape)
        self.states = [alg.State(one_hot[i, :]) for i in range(16)]

        self.graph = random_graph(graph_shape, 0.4)

        layers = [Layer(cortex.Model(), hippocampus.Model())]
        self.model = HUMN(layers)

        explore_steps = 10000
        logging.info("Training a cognitive map:")
        stamp = time.time()
        for i in range(explore_steps):
            path = random_walk(self.graph, random.randint(0, self.graph.shape[0] - 1), self.graph.shape[0] - 1)
            self.model.incrementally_learn([self.states[p] for p in path])
            if i % 100 == 0:
                logging.info(f"Training progress: {(i * 100 / explore_steps):.2f}", end="\r", flush=True)
        logging.info(f"\nFinish learning in {time.time() - stamp}s")

        return Context(model=self.model, states=self.states, goals=np.arange(graph_shape), graph=self.graph, random_seed=self.random_seed)



@contextlib.contextmanager
def experiment_session(path):
    context = Context()
    context.load(path)
    if not context.ready():
        context.setup()
        context.save(path)
    yield context


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    artifact_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "artifacts")
    experiment_path = os.path.join(artifact_path, "experiments", "simple")

    max_steps = 40
    with experiment_session(experiment_path) as context:
        total_length = 0
        stamp = time.time()
        for t in context.goals:
            try:
                path_generator = context.model.find_path(context.states[0], context.states[t], hard_limit=max_steps)
                for pi in path_generator:
                    total_length = total_length + 1
            except RecursionError:
                logging.warning("fail to find path in time.", t)
            finally:
                 logging.info("-----------cognitive planner-----------")
        logging.info("cognitive planner:", time.time() - stamp, " average length:", total_length / len(context.goals))

        total_length = 0
        stamp = time.time()
        for t in context.goals:
            try:
                path_generator = context.model.find_path(context.states[0], context.states[t], hard_limit=max_steps, pathway_bias=-1)
                for pi in path_generator:
                    total_length = total_length + 1
            except RecursionError:
                logging.warning("fail to find path in time.", t)
            finally:
                logging.info("----------hippocampus planner------------")
        logging.info("hippocampus planner:", time.time() - stamp, " average length:", total_length / len(context.goals))
        logging.info("======================================================")

        total_length = 0
        stamp = time.time()
        for t in context.goals:
            try:
                path_generator = context.model.find_path(context.states[0], context.states[t], hard_limit=max_steps, pathway_bias=1)
                for pi in path_generator:
                    total_length = total_length + 1
            except RecursionError:
                logging.warning("fail to find path in time.", t)
            finally:
                logging.info("-----------cortex planner-----------")
        logging.info("cortex planner:", time.time() - stamp, " average length:", total_length / len(context.goals))
        logging.info("======================================================")

        total_length = 0
        stamp = time.time()
        for t in context.goals:
            try:
                p = shortest_path(context.graph, 0, t)
                p = list(reversed(p))
                for pi in p:
                    total_length = total_length + 1
            except RecursionError:
                logging.info("fail to find path in time.", t)
            finally:
                logging.info(p)
                logging.info("-----------optimal planner-----------")

        logging.info("optimal planner:", time.time() - stamp, " average length:", total_length / len(context.goals))

