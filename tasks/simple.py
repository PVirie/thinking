import os
import logging
import contextlib
import random
import json
from typing import List, Any
from pydantic import BaseModel
import argparse
import sys
import math
import jax
import jax.numpy as jnp
from functools import partial

from utilities.utilities import *

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from humn import *
from implementations.jax_onehot import algebraic as alg
from implementations.jax_onehot import cortex, hippocampus, abstraction
import core
from core import table, linear, transformer, stat_head, stat_linear, stat_table


# check --clear flag (default False)
parser = argparse.ArgumentParser()
parser.add_argument("--clear", action="store_true")
args = parser.parse_args()


class Context(BaseModel):
    parameter_sets: List[Any]

    graph_shape: int
    random_seed: int
    num_layers: int
    skip_step_size: int

    states: Any
    goals: Any
    graph: Any
    
    training_state: int = 0

    @staticmethod
    def load(path):
        try:
            with open(os.path.join(path, "metadata.json"), "r") as f:
                metadata = json.load(f)
                parameter_sets = []
                for j in range(len(metadata["parameter_sets"])):
                    set_metadata = metadata["parameter_sets"][j]
                    set_path = os.path.join(path, f"parameter_set_{j}")
                    cortex_models = []
                    hippocampus_models = []
                    for i in range(set_metadata["num_layers"]):
                        layer_path = os.path.join(set_path, "layers", f"layer_{i}")
                        cortex_path = os.path.join(layer_path, "cortex")
                        hippocampus_path = os.path.join(layer_path, "hippocampus")
                        cortex_models.append(cortex.Model.load(cortex_path))
                        hippocampus_models.append(hippocampus.Model.load(hippocampus_path))
                    abstraction_models = []
                    for i in range(set_metadata["num_abstraction_models"]):
                        abstraction_path = os.path.join(set_path, "abstraction_models", f"abstraction_{i}")
                        abstraction_model = None
                        if os.path.exists(abstraction_path):
                            abstraction_model = abstraction.Model.load(abstraction_path)
                        abstraction_models.append(abstraction_model)
                    parameter_sets.append({
                        "cortex_models": cortex_models,
                        "hippocampus_models": hippocampus_models,
                        "abstraction_models": abstraction_models,
                        "name": set_metadata["name"]
                    })
                states = alg.State_Sequence.load(os.path.join(path, "states"))
                goals = np.load(os.path.join(path, "goals.npy"))
                graph = np.load(os.path.join(path, "graph.npy"))
                context = Context(
                    parameter_sets=parameter_sets,
                    graph_shape=metadata["graph_shape"],
                    random_seed=metadata["random_seed"],
                    num_layers=metadata["num_layers"],
                    skip_step_size=metadata["skip_step_size"],
                    states=states, 
                    goals=goals, 
                    graph=graph,
                    training_state=metadata.get("training_state", 0)
                )
            return context
        except Exception as e:
            logging.warning(e)
        return None
                                 

    @staticmethod
    def save(self, path):
        os.makedirs(path, exist_ok=True)
        for j, parameter_set in enumerate(self.parameter_sets):
            set_path = os.path.join(path, f"parameter_set_{j}")
            for i, (c, h) in enumerate(zip(parameter_set["cortex_models"], parameter_set["hippocampus_models"])):
                layer_path = os.path.join(set_path, "layers", f"layer_{i}")
                cortex_path = os.path.join(layer_path, "cortex")
                hippocampus_path = os.path.join(layer_path, "hippocampus")
                cortex.Model.save(c, cortex_path)
                hippocampus.Model.save(h, hippocampus_path)
            for i, model in enumerate(parameter_set["abstraction_models"]):
                if model is None:
                    continue
                abstraction_path = os.path.join(set_path, "abstraction_models", f"abstraction_{i}")
                abstraction.Model.save(model, abstraction_path)
        alg.State_Sequence.save(self.states, os.path.join(path, "states"))
        goal_path = os.path.join(path, "goals.npy")
        np.save(goal_path, self.goals)
        graph_path = os.path.join(path, "graph.npy")
        np.save(graph_path, self.graph)
        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump({
                "parameter_sets": [
                    {
                        "num_layers": len(parameter_set["cortex_models"]),
                        "num_abstraction_models": len(parameter_set["abstraction_models"]),
                        "name": parameter_set["name"]
                    }
                    for parameter_set in self.parameter_sets
                ],
                "graph_shape": self.graph_shape,
                "random_seed": self.random_seed,
                "num_layers": self.num_layers,
                "skip_step_size": self.skip_step_size,
                "training_state": self.training_state
            }, f, indent=4)
        return True


def setup():
    
    graph_shape = 32
    num_layers = 3
    skip_step_size = int(4)
    random_seed = random.randint(0, 1000000)

    one_hot = generate_onehot_representation(np.arange(graph_shape), graph_shape)
    states = alg.State_Sequence(one_hot)

    graph = random_graph(graph_shape, 0.2)

    return Context(
        parameter_sets=[],
        graph_shape=graph_shape,
        random_seed=random_seed,
        num_layers=num_layers,
        skip_step_size=skip_step_size,
        states=states, 
        goals=np.arange(graph_shape), 
        graph=graph)


def train(context, parameter_path):

    graph_shape = context.graph_shape
    random_seed = context.random_seed
    num_layers = context.num_layers
    skip_step_size = context.skip_step_size

    random.seed(random_seed)
    
    explore_steps = 5000
    path_sequences = []
    for i in range(explore_steps):
        path = random_walk(context.graph, math.floor(32 * i / explore_steps), context.graph.shape[0] - 1)
        path_sequences.append(alg.Pointer_Sequence(path))

    def loop_train(trainers, num_epoch=1000):
        print_steps = max(1, num_epoch // 100)
        stamp = time.time()
        for i in range(num_epoch):
            for trainer in trainers:
                trainer.step_update()
            if i % print_steps == 0 and i > 0:
                # print at every 1 % progress
                # compute time to finish in seconds
                logging.info(f"Training progress: {(i * 100 / num_epoch):.2f}, time to finish: {((time.time() - stamp) * (num_epoch - i) / i):.2f}s")
                logging.info(f"Layer loss: {'; '.join([f'{i}| {trainer.avg_loss:.4f}' for i, trainer in enumerate(trainers)])}")
        logging.info(f"Total learning time {time.time() - stamp}s")

    ############################# PREPARE STEP HIERARCHY DATA ################################

    data_skip_path = []
    for p_seq in path_sequences:
        path = context.states.generate_subsequence(p_seq)
        distances = alg.Distance_Sequence(jnp.arange(len(path), dtype=jnp.float32))
        layer_paths = []
        for i in range(num_layers):
            if i == num_layers - 1:
                pivot_indices, pivots = path.sample_skip(math.inf)
                layer_paths.append((path, pivot_indices, distances))
            else:
                pivot_indices, pivots = path.sample_skip(skip_step_size)
                layer_paths.append((path, pivot_indices, distances))
            path = pivots
            distances = distances[pivot_indices.data]
        data_skip_path.append(layer_paths)

    ############################# PREPARE ENTROPIC HIERARCHY DATA ################################

    if context.training_state <= 2:
        abstractor = abstraction.Model(stat_table.Model(graph_shape))
        
        logging.info(f"Learning abstraction")
        for p_seq in path_sequences:
            trainer = abstractor.incrementally_learn(context.states.generate_subsequence(p_seq))
        # for table model, sequential update is neccessary
        trainer.prepare_batch(1)
        loop_train([trainer], explore_steps)
    else:
        abstractor = context.parameter_sets[1]["abstraction_models"][0]

    data_abstract_path = []
    for p_seq in path_sequences:
        path = context.states.generate_subsequence(p_seq)
        distances = alg.Distance_Sequence(jnp.arange(len(path), dtype=jnp.float32))
        layer_paths = []
        for i in range(num_layers):
            if i == num_layers - 1:
                pivot_indices, pivots = path.sample_skip(math.inf)
                layer_paths.append((path, pivot_indices, distances))
            else:
                pivot_indices, pivots = abstractor.abstract_path(path)
                layer_paths.append((path, pivot_indices, distances))
            path = pivots
            distances = distances[pivot_indices.data]
        data_abstract_path.append(layer_paths)

    # print entropy
    logging.info(abstractor.model.infer(context.states.data))

    ############################# PREPARE OPTIMAL DEMONSTRATION ################################

    optimal_path_sequences = []
    for i in range(graph_shape):
        for j in range(graph_shape):
            if i == j:
                continue
            path = shortest_path(context.graph, i, j)
            path = list(reversed(path))
            optimal_path_sequences.append(alg.Pointer_Sequence(path))

    # also add random sequence for noise, five times the number of optimal paths
    explore_steps = len(optimal_path_sequences) * 4
    for i in range(explore_steps):
        path = random_walk(context.graph, math.floor(32 * i / explore_steps), context.graph.shape[0] - 1)
        optimal_path_sequences.append(alg.Pointer_Sequence(path))

    # shuffle
    # current implement of one-hot has issued when optimal values come after random values
    # it may cause zero values in the one-hot representation due to adding raw difference between from and to one-hot vectors

    # random.shuffle(optimal_path_sequences)

    data_optimal_skip_path = []
    for p_seq in optimal_path_sequences:
        path = context.states.generate_subsequence(p_seq)
        distances = alg.Distance_Sequence(jnp.arange(len(path), dtype=jnp.float32))
        layer_paths = []
        for i in range(num_layers):
            if i == num_layers - 1:
                pivot_indices, pivots = path.sample_skip(math.inf)
                layer_paths.append((path, pivot_indices, distances))
            else:
                pivot_indices, pivots = path.sample_skip(skip_step_size)
                layer_paths.append((path, pivot_indices, distances))
            path = pivots
            distances = distances[pivot_indices.data]
        data_optimal_skip_path.append(layer_paths)


    ############################# TRAINING ################################

    ############################# SET 1 ################################
    if context.training_state <= 1:

        name = "Skip steps"

        # For table experiment, hidden_size is the crucial parameter.
        cortex_models = [
            cortex.Model(i, transformer.Model(graph_shape, 1, 128, [128], memory_size=graph_shape, value_access=False, lr=0.001, r_seed=random_seed), max_steps = graph_shape)
            for i in range(num_layers)
        ]

        hippocampus_models = [
            hippocampus.Model(graph_shape, graph_shape)
            for i in range(num_layers)
        ]

        abstraction_models = []

        logging.info(f"Training experiment {name}")
        
        model = HUMN(cortex_models, hippocampus_models, abstraction_models)

        for path_tuples in data_skip_path:
            trainers = model.observe(path_tuples)
        for trainer in trainers:
            trainer.prepare_batch(32)
        loop_train(trainers, 100000)

        context.parameter_sets.append({
            "cortex_models": cortex_models,
            "hippocampus_models": hippocampus_models,
            "abstraction_models": abstraction_models,
            "name": name
        })
        context.training_state = 2
        random_seed = random.randint(0, 1000000)
        context.random_seed = random_seed
        Context.save(context, parameter_path)


    ############################# SET 2 ################################
    if context.training_state <= 2:

        name = "Entropy"

        # For table experiment, hidden_size is the crucial parameter.
        cortex_models = [
            cortex.Model(i, transformer.Model(graph_shape, 1, 128, [128], memory_size=graph_shape, value_access=False, lr=0.001, r_seed=random_seed), max_steps = graph_shape)
            for i in range(num_layers)
        ]

        hippocampus_models = [
            hippocampus.Model(graph_shape, graph_shape)
            for i in range(num_layers)
        ]
        
        abstraction_models = []
        for i in range(len(cortex_models) - 1):
            abstraction_models.append(abstractor)

        logging.info(f"Training experiment {name}")

        model = HUMN(cortex_models, hippocampus_models, abstraction_models)

        for path_tuples in data_abstract_path:
            trainers = model.observe(path_tuples)
        for trainer in trainers:
            trainer.prepare_batch(32)
        loop_train(trainers, 100000)

        context.parameter_sets.append({
            "cortex_models": cortex_models,
            "hippocampus_models": hippocampus_models,
            "abstraction_models": abstraction_models,
            "name": name
        })
        context.training_state = 3
        random_seed = random.randint(0, 1000000)
        context.random_seed = random_seed
        Context.save(context, parameter_path)


    ############################# SET 3 ################################
    if context.training_state <= 3:

        name = "Optimal"

        table_cores = []
        for i in range(num_layers):
            table_core = table.Model(graph_shape, 1)
            table_cores.append(table_core)

        cortex_models = []
        hippocampus_models = []
        for i in range(num_layers):
            c = cortex.Model(i, table_cores[i], max_steps = graph_shape)
            cortex_models.append(c)

            h = hippocampus.Model(graph_shape, graph_shape)
            hippocampus_models.append(h)

        abstraction_models = []

        logging.info(f"Training experiment {name}")

        model = HUMN(cortex_models, hippocampus_models, abstraction_models)

        for path_tuples in data_optimal_skip_path:
            trainers = model.observe(path_tuples)
        for trainer in trainers:
            # for table model, sequential update is neccessary
            trainer.prepare_batch(1)
        loop_train(trainers, len(data_optimal_skip_path))

        context.parameter_sets.append({
            "cortex_models": cortex_models,
            "hippocampus_models": hippocampus_models,
            "abstraction_models": abstraction_models,
            "name": "Table layers"
        })
        context.training_state = 4
        random_seed = random.randint(0, 1000000)
        context.random_seed = random_seed
        Context.save(context, parameter_path)


@contextlib.contextmanager
def experiment_session(path, force_clear=None):
    core.initialize(os.path.join(path, "core"))
    if force_clear is not None and force_clear:
        logging.info(f"Clearing the experiment directory: {path}")
        empty_directory(path)
    context = Context.load(path)
    if context is None:
        context = setup()
    train(context, path)
    yield context


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    experiment_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "experiments", "simple")

    if args.clear:
        # just clear and return
        empty_directory(experiment_path)
        exit(0)

    with experiment_session(experiment_path) as context:
        
        max_steps = len(context.states)

        def exp_loop(model, think_ahead=False):
            total_length = 0
            stamp = time.time()
            for t_i in context.goals:
                s = context.states[0]
                t = context.states[t_i]
                ps = [0]
                model.refresh()
                
                if think_ahead:
                    try:
                        for p in model.think(s, t - s):
                            p_i = context.states[p]
                            ps.append(p_i)
                        total_length = total_length + len(ps)
                    except MaxSubStepReached:
                        logging.warning("fail to find path in time.")
                        total_length = total_length + max_steps
                else:
                    for i in range(max_steps):
                        if s == t:
                            break
                        a = model.react(s, t - s)
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

        for i, parameter_set in enumerate(context.parameter_sets):
            # if i == 0:
            #     for j, c in enumerate(parameter_set["cortex_models"]):
            #         def printer(level, s):
            #             s_i = context.states[s]
            #             logging.info(f"Level: {level} State: {s_i}")
            #         c.printer = partial(printer, j)

            logging.info(f"-----------cognitive planner {parameter_set['name']}-----------")
            total_length, elapsed_seconds = exp_loop(HUMN(**parameter_set), think_ahead=False)
            logging.info(f"cognitive planner (reactive) {parameter_set['name']}: {elapsed_seconds:.2f}s, average length: {total_length / len(context.goals):.2f}")

            total_length, elapsed_seconds = exp_loop(HUMN(**parameter_set), think_ahead=True)
            logging.info(f"cognitive planner (think ahead) {parameter_set['name']}: {elapsed_seconds:.2f}s, average length: {total_length / len(context.goals):.2f}")


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
