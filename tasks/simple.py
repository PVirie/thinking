import os
import logging
import contextlib
import random
import json
from typing import List, Any
import jax.numpy
from pydantic import BaseModel
from functools import partial
import argparse
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from humn import *
from src.utilities import *
from src.jax import algebric as alg
from src.jax import cortex, hippocampus, abstraction
import src.core as core
from src.core import table, linear, transformer, stat_head, stat_linear
import jax


# check --clear flag (default False)
parser = argparse.ArgumentParser()
parser.add_argument("--clear", action="store_true")
args = parser.parse_args()


class Context(BaseModel):
    parameter_sets: List[Any]
    states: Any
    goals: Any
    graph: Any
    random_seed: int = 0

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
                context = Context(parameter_sets=parameter_sets, abstraction_models=abstraction_models, states=states, goals=goals, graph=graph, random_seed=metadata["random_seed"])
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
        explore_steps = 1000
        path_sequences = []
        for i in range(explore_steps):
            path = random_walk(graph, 0, graph.shape[0] - 1)
            path_sequences.append(alg.Pointer_Sequence(path))

        data_skip_path = []
        max_layers = 4
        for p_seq in path_sequences:
            path = states.generate_subsequence(p_seq)
            layer_paths = []
            for i in range(max_layers):
                if i == max_layers - 1:
                    pivot_indices, pivots = path.sample_skip(math.inf)
                    layer_paths.append((path, pivot_indices, pivots))
                else:
                    pivot_indices, pivots = path.sample_skip(2)
                    layer_paths.append((path, pivot_indices, pivots))
                path = pivots
            data_skip_path.append(layer_paths)


        for datum in data_skip_path:
            path = datum[-1][0]
            print("-", jax.numpy.argmax(path.data, axis=1))
            pivots = datum[-1][2]
            print(">", jax.numpy.argmax(pivots.data, axis=1))


        # abstractor = abstraction.Model(stat_linear.Model(32, graph_shape))

        # logging.info(f"Learning abstraction")
        # num_epoch = 10000
        # stamp = time.time()
        # for i in range(num_epoch):
        #     p_seq = path_sequences[i % len(path_sequences)]
        #     abstractor.incrementally_learn(states.generate_subsequence(p_seq))
        #     if i % print_steps == 0 and i > 0:
        #         # print at every 1 % progress
        #         # compute time to finish in seconds
        #         logging.info(f"Training progress: {(i * 100 / num_epoch):.2f}, time to finish: {((time.time() - stamp) * (num_epoch - i) / i):.2f}s")
        # logging.info(f"Total learning time {time.time() - stamp}s")
        
        # data_abstract_path = []
        # max_layers = 3
        # for p_seq in path_sequences:
        #     path = states.generate_subsequence(p_seq)
        #     layer_paths = []
        #     for i in range(max_layers):
        #         if i == max_layers - 1:
        #             pivot_indices, pivots = path.sample_skip(math.inf)
        #             layer_paths.append((path, pivot_indices, pivots))
        #         else:
        #             pivot_indices, pivots = abstractor.abstract_path(path)
        #             layer_paths.append((path, pivot_indices, pivots))
        #         path = pivots
        #     data_abstract_path.append(layer_paths)


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
                    logging.info(f"Layer loss: {', '.join([f'{trainer.avg_loss:.4f}' for trainer in trainers])}")
            logging.info(f"Total learning time {time.time() - stamp}s")


        parameter_sets = []
        ############################# SET 1 ################################

        name = "Skip step"

        cortex_models = [
            cortex.Model(0, linear.Model(64, graph_shape)),
            cortex.Model(1, linear.Model(64, graph_shape)),
            cortex.Model(2, linear.Model(32, graph_shape))
        ]
        hippocampus_models = [
            hippocampus.Model(graph_shape, graph_shape),
            hippocampus.Model(graph_shape, graph_shape),
            hippocampus.Model(graph_shape, graph_shape)
        ]
        abstraction_models = []
        
        model = HUMN(cortex_models, hippocampus_models, abstraction_models)

        logging.info(f"Training experiment {name}")

        for path_tuples in data_skip_path:
            trainers = model.observe(path_tuples)
        for trainer in trainers:
            trainer.prepare_batch(32)

        loop_train(trainers, 5000)

        parameter_sets.append({
            "cortex_models": cortex_models,
            "hippocampus_models": hippocampus_models,
            "abstraction_models": abstraction_models,
            "name": name
        })

        # ############################# SET 2 ################################

        # name = "Entropy 3 layers"

        # linear_cores = []
        # for i in range(3):
        #     linear_core = linear.Model(graph_shape, graph_shape)
        #     linear_cores.append(linear_core)

        # cortex_models = []
        # hippocampus_models = []
        # for i in range(3):
        #     c = cortex.Model(i, linear_cores[i])
        #     cortex_models.append(c)

        #     h = hippocampus.Model(graph_shape, graph_shape)
        #     hippocampus_models.append(h)
            

        # abstraction_models = []
        # for i in range(2):
        #     abstraction_models.append(abstractor)

        # model = HUMN(cortex_models, hippocampus_models, abstraction_models)

        # logging.info(f"Training experiment {name}")

        # for path_tuples in data_abstract_path:
        #     trainers = model.observe(path_tuples)
        # for trainer in trainers:
        #     trainer.prepare_batch(32)

        # loop_train(trainers)

        # parameter_sets.append({
        #     "cortex_models": cortex_models,
        #     "hippocampus_models": hippocampus_models,
        #     "abstraction_models": abstraction_models,
        #     "name": name
        # })

        ############################# SET 3 ################################

        name = "Table layers"

        table_cores = []
        for i in range(3):
            table_core = table.Model(graph_shape)
            table_cores.append(table_core)

        cortex_models = []
        hippocampus_models = []
        for i in range(3):
            c = cortex.Model(i, table_cores[i])
            cortex_models.append(c)

            h = hippocampus.Model(graph_shape, graph_shape)
            hippocampus_models.append(h)

        abstraction_models = []

        model = HUMN(cortex_models, hippocampus_models, abstraction_models)

        logging.info(f"Training experiment {name}")

        for path_tuples in data_skip_path:
            trainers = model.observe(path_tuples)
        for trainer in trainers:
            # for table model, sequential update is neccessary
            trainer.prepare_batch(1)

        loop_train(trainers, 1000)

        parameter_sets.append({
            "cortex_models": cortex_models,
            "hippocampus_models": hippocampus_models,
            "abstraction_models": abstraction_models,
            "name": "Table layers"
        })

        return Context(parameter_sets=parameter_sets, abstraction_models=abstraction_models, states=states, goals=np.arange(graph_shape), graph=graph, random_seed=random_seed)



@contextlib.contextmanager
def experiment_session(path, force_clear=None):
    core.initialize(os.path.join(path, "core"))
    if force_clear is not None and force_clear:
        logging.info(f"Clearing the experiment directory: {path}")
        empty_directory(path)
    context = Context.load(path)
    if context is None:
        context = Context.setup(path)
        Context.save(context, path)
    yield context


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    experiment_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "experiments", "simple")

    if args.clear:
        # just clear and return
        empty_directory(experiment_path)
        exit(0)

    max_steps = 40
    with experiment_session(experiment_path) as context:

        def exp_loop(model):
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
                    a = model.infer_sub_action(s, t - s)
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
            total_length, elapsed_seconds = exp_loop(HUMN(**parameter_set))
            logging.info(f"cognitive planner {parameter_set['name']}: {elapsed_seconds:.2f}s, average length: {total_length / len(context.goals):.2f}")

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
