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
import jax.numpy

from array2gif import write_gif
import gym

# replace np.bool8 with np.bool
import numpy as np
np.bool8 = np.bool

from utilities.utilities import *

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from humn import *
from implementations.jax_cart_pole import algebric as alg
from implementations.jax_cart_pole import cortex, hippocampus, abstraction
import core
from core import table, linear, transformer, stat_head, stat_linear, stat_table


# check --clear flag (default False)
parser = argparse.ArgumentParser()
parser.add_argument("--clear", action="store_true")
args = parser.parse_args()


class Context(BaseModel):
    parameter_sets: List[Any]
    average_random_steps: float
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
                context = Context(parameter_sets=parameter_sets, abstraction_models=abstraction_models, average_random_steps=metadata["average_random_steps"], random_seed=metadata["random_seed"])
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
                "average_random_steps": self.average_random_steps,
                "random_seed": self.random_seed
            }, f, indent=4)
        return True


    @staticmethod
    def setup(setup_path):
        random_seed = random.randint(0, 1000)
        random.seed(random_seed)

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
                    logging.info(f"Layer loss: {'| '.join([f'{i}, {trainer.avg_loss:.4f}' for i, trainer in enumerate(trainers)])}")
            logging.info(f"Total learning time {time.time() - stamp}s")


        def prepare_data_tuples(states, actions, rewards, num_layers, skip_steps):
            states = np.stack(states, axis=0)
            actions = np.reshape(np.stack(actions, axis=0), (-1, 1))
            rewards = np.reshape(np.stack(rewards, axis=0), (-1, 1))
            path_layer_tuples = [] # List[Tuple[algebraic.State_Action_Sequence, algebraic.Pointer_Sequence, algebraic.Expectation_Sequence]]
            for layer_i in range(num_layers):
                path = alg.State_Action_Sequence(np.concatenate([states, actions], axis=1))

                skip_sequence = [i for i in range(0, len(states), skip_steps)]
                # always add the last index
                if skip_sequence[-1] != len(states) - 1:
                    skip_sequence.append(len(states) - 1)

                skip_pointer_sequence = alg.Pointer_Sequence(skip_sequence)
                expectation_sequence = alg.Expectation_Sequence(states[skip_sequence, :])
                path_layer_tuples.append((path, skip_pointer_sequence, expectation_sequence))

                states = states[skip_sequence]
                actions = actions[skip_sequence]
                rewards = rewards[skip_sequence]
            
            return path_layer_tuples


        parameter_sets = []
        ############################# SET 1 ################################

        name = "Skip step"

        state_dim = 4
        action_dim = 1
        expectation_dim = 4
        context_length = 1

        cortex_models = [
            cortex.Model(0, return_action=True, model=transformer.Model([state_dim, action_dim, expectation_dim], context_length, 128, [128, 64], memory_size=4, lr=0.001, r_seed=random_seed)),
            cortex.Model(1, return_action=False, model=transformer.Model([state_dim, expectation_dim, expectation_dim], context_length, 128, [128, 64], memory_size=4, lr=0.001, r_seed=random_seed)),
            cortex.Model(2, return_action=False, model=transformer.Model([state_dim, expectation_dim, expectation_dim], context_length, 128, [128, 64], memory_size=4, lr=0.001, r_seed=random_seed)),
        ]

        hippocampus_models = [
            hippocampus.Model(state_dim),
            hippocampus.Model(state_dim),
            hippocampus.Model(state_dim)
        ]

        abstraction_models = []
        model = HUMN(cortex_models, hippocampus_models, abstraction_models)

        logging.info(f"Training experiment {name}")

        env = gym.make("CartPole-v1", render_mode=None)
        env.action_space.seed(random_seed)
        observation, info = env.reset(seed=random_seed)

        skip_steps = 8
        num_layers = len(cortex_models)
        
        total_steps = 0
        num_trials = 10000
        print_steps = max(1, num_trials // 100)
        for i in range(num_trials):
            if i % print_steps == 0 and i > 0:
                # print at every 1 % progress
                logging.info(f"Environment collection: {(i * 100 / num_trials):.2f}")

            states = []
            actions = []
            rewards = []
            for _ in range(1000):
                selected_action = env.action_space.sample()
                next_observation, reward, terminated, truncated, info = env.step(selected_action)
                states.append(observation)
                actions.append(selected_action)
                rewards.append(reward)
                if terminated or truncated:
                    observation, info = env.reset()
                    break
                else:
                    observation = next_observation
            total_steps += len(states)

            # now make hierarchical data
            path_layer_tuples = prepare_data_tuples(states, actions, rewards, num_layers, skip_steps)
            trainers = model.observe(path_layer_tuples)

        env.close()

        for trainer in trainers:
            trainer.prepare_batch(16)

        loop_train(trainers, 20000)

        parameter_sets.append({
            "cortex_models": cortex_models,
            "hippocampus_models": hippocampus_models,
            "abstraction_models": abstraction_models,
            "name": name
        })

        previous_model = model

        ############################# SET 2 ################################

        name = "Skip step (Round 2)"

        state_dim = 4
        action_dim = 1
        expectation_dim = 4
        context_length = 1

        cortex_models = [
            cortex.Model(0, return_action=True, model=transformer.Model([state_dim, action_dim, expectation_dim], context_length, 128, [128, 64], memory_size=4, lr=0.001, r_seed=random_seed)),
            cortex.Model(1, return_action=False, model=transformer.Model([state_dim, expectation_dim, expectation_dim], context_length, 128, [128, 64], memory_size=4, lr=0.001, r_seed=random_seed)),
            cortex.Model(2, return_action=False, model=transformer.Model([state_dim, expectation_dim, expectation_dim], context_length, 128, [128, 64], memory_size=4, lr=0.001, r_seed=random_seed)),
        ]

        hippocampus_models = [
            hippocampus.Model(state_dim),
            hippocampus.Model(state_dim),
            hippocampus.Model(state_dim)
        ]

        abstraction_models = []
        model = HUMN(cortex_models, hippocampus_models, abstraction_models)

        logging.info(f"Training experiment {name}")

        env = gym.make("CartPole-v1", render_mode=None)
        env.action_space.seed(random_seed)
        observation, info = env.reset(seed=random_seed)
        stable_state = alg.Expectation([0, 0, 0, 0])

        skip_steps = 8
        num_layers = len(cortex_models)
        
        total_steps = 0
        num_trials = 10000
        print_steps = max(1, num_trials // 100)
        for i in range(num_trials):
            if i % print_steps == 0 and i > 0:
                # print at every 1 % progress
                logging.info(f"Environment collection: {(i * 100 / num_trials):.2f}")

            states = []
            actions = []
            rewards = []
            for _ in range(1000):
                if random.random() < 0.1:
                    selected_action = env.action_space.sample()
                else:
                    a = previous_model.react(observation, stable_state)
                    selected_action = 1 if np.asarray(a.data)[0].item() > 0.5 else 0

                next_observation, reward, terminated, truncated, info = env.step(selected_action)
            
                states.append(observation)
                actions.append(selected_action)
                rewards.append(reward)
                if terminated or truncated:
                    observation, info = env.reset()
                    break
                else:
                    observation = next_observation
            total_steps += len(states)

            # now make hierarchical data
            path_layer_tuples = prepare_data_tuples(states, actions, rewards, num_layers, skip_steps)
            trainers = model.observe(path_layer_tuples)

        env.close()

        for trainer in trainers:
            trainer.prepare_batch(16)

        loop_train(trainers, 20000)

        parameter_sets.append({
            "cortex_models": cortex_models,
            "hippocampus_models": hippocampus_models,
            "abstraction_models": abstraction_models,
            "name": name
        })

        return Context(parameter_sets=parameter_sets, average_random_steps=total_steps/num_trials, random_seed=random_seed)



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
    experiment_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "experiments", "cart_pole")

    if args.clear:
        # just clear and return
        empty_directory(experiment_path)
        exit(0)

    with experiment_session(experiment_path) as context:
        
        logging.info(f"Baseline number of random steps: {context.average_random_steps}")

        env = gym.make("CartPole-v1", render_mode="rgb_array")
        env.action_space.seed(random.randint(0, 1000))
        observation, info = env.reset(seed=random.randint(0, 1000))

        def generate_visual(render_path, num_trials, action_method):
            observation, info = env.reset()
            for j in range(5):
                output_gif = os.path.join(render_path, f"trial_{j}.gif")
                imgs = []
                for _ in range(1000):
                    selected_action = action_method(observation)
                    observation, reward, terminated, truncated, info = env.step(selected_action)
                    img = env.render()
                    imgs.append(np.transpose(img, [2, 0, 1]))
                    if terminated or truncated:
                        observation, info = env.reset()
                        break
                write_gif(imgs, output_gif, fps=30)


        logging.info("----------------- Testing random behavior -----------------")
        observation, info = env.reset()
        result_path = os.path.join(experiment_path, "results", f"random")
        render_path = os.path.join(result_path, "render")
        os.makedirs(render_path, exist_ok=True)
        generate_visual(render_path, 10, lambda obs: env.action_space.sample())


        for i, parameter_set in enumerate(context.parameter_sets):
            result_path = os.path.join(experiment_path, "results", f"set_{i}")

            model = HUMN(**parameter_set)
            observation, info = env.reset()
            stable_state = alg.Expectation([0, 0, 0, 0])

            total_steps = 0
            num_trials = 100
            print_steps = max(1, num_trials // 10)
            for j in range(num_trials):
                if j % print_steps == 0 and j > 0:
                    # print at every 1 % progress
                    logging.info(f"Environment testing: {(j * 100 / num_trials):.2f}")
                for _ in range(1000):
                    # selected_action = env.action_space.sample()
                    a = model.react(observation, stable_state)
                    selected_action = 1 if np.asarray(a.data)[0].item() > 0.5 else 0
                    observation, reward, terminated, truncated, info = env.step(selected_action)
                    total_steps += 1
                    if terminated or truncated:
                        observation, info = env.reset()
                        break

            set_name = parameter_set["name"]
            logging.info(f"Parameter set {set_name}; average behavior steps: {total_steps/num_trials}")

            render_path = os.path.join(result_path, "render")
            os.makedirs(render_path, exist_ok=True)

            def generation_action(observation):
                a = model.react(observation, stable_state)
                return 1 if np.asarray(a.data)[0].item() > 0.5 else 0
            
            generate_visual(render_path, 10, generation_action)

        env.close()
