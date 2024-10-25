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

import gymnasium as gym

# replace np.bool8 with np.bool
import numpy as np
np.bool8 = np.bool

from utilities.utilities import *

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from humn import *
from implementations.jax_rl import algebraic as alg
from implementations.jax_rl import cortex, hippocampus, abstraction
import core
from core import table, linear, transformer, stat_head, stat_linear, stat_table


# check --clear flag (default False)
parser = argparse.ArgumentParser()
parser.add_argument("--clear", action="store_true")
args = parser.parse_args()


class Context(BaseModel):
    parameter_sets: List[Any]
    goals: List[Tuple[Any, str]] = []
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
                context = Context(parameter_sets=parameter_sets, goals=metadata["goals"], random_seed=metadata["random_seed"])
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
                "goals": self.goals,
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


        def states_to_expectation(states, rewards):
            vx = states[:, 5]
            vz = np.abs(states[:, 6])
            return np.stack([rewards, vx, vz], axis=1)


        def compute_goal_diff(base, operand):
            # base has shape (n, dim), operand has shape (m, dim)
            base_ = np.tile(np.expand_dims(base, axis=1), (1, operand.shape[0], 1))
            operand_ = np.tile(np.expand_dims(operand, axis=0), (base.shape[0], 1, 1))
            raw_diff = (base_ - operand_) ** 2
            diff = np.sum(raw_diff, axis=-1, keepdims=False)
            return diff

        goals = [
            ([5, 5], "jump forward"),
            ([0, 5], "jump still"),
            ([-5, 5], "jump backward"),
        ]

        def update_best_so_far(last_pivots, best_targets, best_target_diffs):

            last_goals = last_pivots[:, 1:]
            diffs = compute_goal_diff(np.array([g[0] for g in goals]), last_goals)
            # diffs has shape (num_goals, num_states)
            min_indices = np.argmin(diffs, axis=1, keepdims=True)
            min_scores = np.take_along_axis(diffs, min_indices, axis=1)
            min_goals = np.take_along_axis(last_goals, min_indices, axis=0)
            update_flags = min_scores < best_target_diffs

            best_target_diffs = np.where(update_flags, min_scores, best_target_diffs)
            best_targets = np.where(update_flags, min_goals, best_targets)

            return best_targets, best_target_diffs


        def prepare_data_tuples(states, actions, rewards, num_layers, skip_steps):
            states = np.stack(states, axis=0)
            actions = np.stack(actions, axis=0)
            expectations = states_to_expectation(states, rewards)
            path_layer_tuples = [] # List[Tuple[algebraic.State_Action_Sequence, algebraic.Pointer_Sequence, algebraic.Expectation_Sequence]]
            for layer_i in range(num_layers):
                path = alg.State_Action_Sequence(states, actions)

                skip_sequence = [i for i in range(0, len(states), skip_steps)]
                # always add the last index
                if skip_sequence[-1] != len(states) - 1:
                    skip_sequence.append(len(states) - 1)
                skip_pointer_sequence = alg.Pointer_Sequence(skip_sequence)

                if layer_i == num_layers - 1:
                    last_pivots = expectations[skip_sequence, :]
                    expectation_sequence = alg.Expectation_Sequence(last_pivots)
                else:
                    expectation_sequence = alg.Expectation_Sequence(expectations[skip_sequence, 0:1], states[skip_sequence, :])
                    
                path_layer_tuples.append((path, skip_pointer_sequence, expectation_sequence))

                states = states[skip_sequence]
                actions = actions[skip_sequence]
                # expectation is the average of the expectations in the skip sequence
                expectations = compute_sum_along_sequence(expectations, skip_sequence) / skip_steps
            
            return path_layer_tuples, last_pivots


        parameter_sets = []
        ############################# SET 1 ################################

        name = "Curriculum (Skip steps)"
        skip_steps = 4

        state_dim = 11
        action_dim = 3
        expectation_dim = 2 # acc_x, speed_z
        context_length = 1

        cortex_models = [
            cortex.Model(0, return_action=True, use_reward=False, model=transformer.Model([state_dim, action_dim, state_dim], context_length, 128, [128, 64], memory_size=16, lr=0.0001, r_seed=random_seed)),
            cortex.Model(1, return_action=False, use_reward=False, model=transformer.Model([state_dim, state_dim, state_dim], context_length, 256, [256, 128], memory_size=16, lr=0.0001, r_seed=random_seed)),
            cortex.Model(2, return_action=False, use_reward=False, model=transformer.Model([state_dim, state_dim, state_dim], context_length, 256, [256, 128, 128], memory_size=16, lr=0.0001, r_seed=random_seed)),
            cortex.Model(3, return_action=False, use_reward=False, model=transformer.Model([state_dim, state_dim, expectation_dim], context_length, 256, [256, 256, 128, 128], memory_size=16, lr=0.0001, r_seed=random_seed)),
        ]

        hippocampus_models = [
            hippocampus.Model(state_dim),
            hippocampus.Model(state_dim),
            hippocampus.Model(state_dim),
            hippocampus.Model(state_dim)
        ]

        abstraction_models = []
        model = HUMN(cortex_models, hippocampus_models, abstraction_models)

        logging.info(f"Training experiment {name}")

        env = gym.make(
            'Hopper-v5',
            healthy_reward=1, forward_reward_weight=0, ctrl_cost_weight=1e-3, 
            healthy_angle_range=(-math.pi / 2, math.pi / 2), healthy_state_range=(-100, 100), 
            render_mode=None
        )
        env.action_space.seed(random_seed)
        observation, info = env.reset(seed=random_seed)

        num_layers = len(cortex_models)
        num_courses = 20
        for course in range(num_courses):
            logging.info(f"Course {course}")
            total_steps = 0
            num_trials = 2000
            print_steps = max(1, num_trials // 100)
            epsilon = 0.2 + 0.7 * (course + 1) / num_courses

            next_best_targets = np.zeros((len(goals), len(goals[0][0])), dtype=np.float32)
            next_best_target_diffs = np.ones((len(goals), 1), dtype=np.float32) * 1e4

            for i in range(num_trials):
                if i % print_steps == 0 and i > 0:
                    # print at every 1 % progress
                    logging.info(f"Environment collection: {(i * 100 / num_trials):.2f}%")
                
                if course > 0:
                    target = best_targets[i % len(goals), :]
                    stable_state = alg.Expectation(target)
                
                observation, info = env.reset()
                states = []
                actions = []
                rewards = []
                for _ in range(200):
                    if random.random() >= epsilon or course == 0:
                        selected_action = env.action_space.sample()
                    else:
                        a = model.react(alg.State(observation.data), stable_state)
                        selected_action = np.clip(np.asarray(a.data), -1, 1)

                    next_observation, reward, terminated, truncated, info = env.step(selected_action)

                    states.append(observation)
                    actions.append(selected_action)
                    rewards.append(reward)
                    if terminated or truncated:
                        break
                    else:
                        observation = next_observation

                total_steps += len(states)
                # now make hierarchical data
                path_layer_tuples, last_pivots = prepare_data_tuples(states, actions, rewards, num_layers, skip_steps)
                trainers = model.observe(path_layer_tuples)

                next_best_targets, next_best_target_diffs = update_best_so_far(last_pivots, next_best_targets, next_best_target_diffs)

            logging.log(logging.INFO, f"Average steps: {total_steps/num_trials}")
            env.close()

            for trainer in trainers:
                trainer.prepare_batch(max_mini_batch_size=16, max_learning_sequence=32)

            loop_train(trainers, 20000)

            for trainer in trainers:
                trainer.clear_batch()

            best_targets = next_best_targets

        parameter_sets.append({
            "cortex_models": cortex_models,
            "hippocampus_models": hippocampus_models,
            "abstraction_models": abstraction_models,
            "name": name
        })

        return Context(parameter_sets=parameter_sets, goals=goals, random_seed=random_seed)



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
    experiment_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "experiments", "hopper")

    if args.clear:
        # just clear and return
        empty_directory(experiment_path)
        exit(0)

    with experiment_session(experiment_path) as context:
        
        env = gym.make(
            'Hopper-v5',
            healthy_reward=1, forward_reward_weight=0, ctrl_cost_weight=1e-3, 
            healthy_angle_range=(-math.pi / 2, math.pi / 2), healthy_state_range=(-100, 100), 
            render_mode="rgb_array", width=1280, height=720
        )
        env.action_space.seed(random.randint(0, 1000))
        observation, info = env.reset(seed=random.randint(0, 1000))

        def generate_visual(render_path, num_trials, action_method):
            observation, info = env.reset()
            for j in range(num_trials):
                observation, info = env.reset()
                output_gif = os.path.join(render_path, f"trial_{j}.gif")
                imgs = []
                for _ in range(500):
                    selected_action = action_method(observation)
                    observation, reward, terminated, truncated, info = env.step(selected_action)
                    img = env.render()
                    imgs.append(img)
                    if terminated or truncated:
                        break
                write_gif(imgs, output_gif, fps=30)

        for goal, goal_text in context.goals:

            for i, parameter_set in enumerate(context.parameter_sets):
                result_path = os.path.join(experiment_path, "results", goal_text, f"set_{i}")

                model = HUMN(**parameter_set)
                observation, info = env.reset()

                total_steps = 0
                num_trials = 100
                print_steps = max(1, num_trials // 10)
                for j in range(num_trials):
                    if j % print_steps == 0 and j > 0:
                        # print at every 1 % progress
                        logging.info(f"Environment testing: {(j * 100 / num_trials):.2f}")
                    observation, info = env.reset()
                    for _ in range(1000):
                        # selected_action = env.action_space.sample()
                        a = model.react(alg.State(observation.data), alg.Expectation(goal))
                        selected_action = np.clip(np.asarray(a.data), -1, 1)
                        observation, reward, terminated, truncated, info = env.step(selected_action)
                        total_steps += 1
                        if terminated or truncated:
                            break

                set_name = parameter_set["name"]
                logging.info(f"Parameter set {set_name}; average behavior steps: {total_steps/num_trials}")

                render_path = os.path.join(result_path, "render")
                os.makedirs(render_path, exist_ok=True)

                def generation_action(observation):
                    a = model.react(alg.State(observation.data), alg.Expectation(goal))
                    return np.clip(np.asarray(a.data), -1, 1)
                
                generate_visual(render_path, 2, generation_action)

        env.close()
