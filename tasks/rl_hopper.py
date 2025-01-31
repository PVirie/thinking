import os
import logging
import contextlib
import random
import json
from typing import List, Any, Union
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
    cortex_models: Any
    hippocampus_models: Any
    abstraction_models: Any
    name: str
    skip_steps: int
    goals: List[Any]

    random_seed: int = 0
    course: int = 0
    best_goals: Union[None, List[List[float]]] = None

    @staticmethod
    def load(path):
        try:
            with open(os.path.join(path, "metadata.json"), "r") as f:
                set_metadata = json.load(f)
                set_path = os.path.join(path, f"model_parameters")
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
                context = Context(
                    cortex_models=cortex_models,
                    hippocampus_models=hippocampus_models,
                    abstraction_models=abstraction_models,
                    name=set_metadata["name"],
                    skip_steps=set_metadata["skip_steps"],
                    goals=set_metadata["goals"],
                    random_seed=set_metadata["random_seed"],
                    course=set_metadata["course"] if "course" in set_metadata else 0,
                    best_goals=set_metadata["best_goals"] if "best_goals" in set_metadata else None
                )
            return context
        except Exception as e:
            logging.warning(e)
        return None
                                 

    @staticmethod
    def save(self, path):
        os.makedirs(path, exist_ok=True)
        set_path = os.path.join(path, f"model_parameters")
        for i, (c, h) in enumerate(zip(self.cortex_models, self.hippocampus_models)):
            layer_path = os.path.join(set_path, "layers", f"layer_{i}")
            cortex_path = os.path.join(layer_path, "cortex")
            hippocampus_path = os.path.join(layer_path, "hippocampus")
            cortex.Model.save(c, cortex_path)
            hippocampus.Model.save(h, hippocampus_path)
        for i, model in enumerate(self.abstraction_models):
            if model is None:
                continue
            abstraction_path = os.path.join(set_path, "abstraction_models", f"abstraction_{i}")
            abstraction.Model.save(model, abstraction_path)
        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump({
                "num_layers": len(self.cortex_models),
                "num_abstraction_models": len(self.hippocampus_models),
                "name": self.name,
                "skip_steps": self.skip_steps,
                "goals": self.goals,
                "random_seed": self.random_seed,
                "course": self.course,
                "best_goals": self.best_goals
            }, f, indent=4)
        return True


def setup():
    random_seed = random.randint(0, 1000)
    random.seed(random_seed)

    name = "Curriculum (Skip steps)"
    skip_steps = 16

    state_dim = 11
    action_dim = 3
    expectation_dim = 2 # velo_x, speed_z
    context_length = 1

    cortex_models = [
        cortex.Model(0, return_action=True, continuous_reward=False, step_discount_factor=0.98, model=transformer.Model([state_dim, action_dim, state_dim], context_length, 128, [256, 256], memory_size=32, value_access=True, lr=0.0001, r_seed=random_seed)),
        cortex.Model(1, return_action=False, continuous_reward=True, step_discount_factor=0.98, model=transformer.Model([state_dim, state_dim, expectation_dim], context_length, 128, [256, 256], memory_size=64, value_access=True, lr=0.0001, r_seed=random_seed)),
    ]

    hippocampus_models = [
        hippocampus.Model(state_dim),
        hippocampus.Model(state_dim),
        hippocampus.Model(state_dim),
        hippocampus.Model(state_dim)
    ]

    abstraction_models = []

    return Context(
        cortex_models=cortex_models,
        hippocampus_models=hippocampus_models,
        abstraction_models=abstraction_models,
        name=name,
        skip_steps=skip_steps,
        goals=[
            ([3, 1], "jump forward"),
            ([0, 2], "jump still"),
            ([-3, 1], "jump backward"),
        ],
        random_seed=random_seed,
        course=0,
        best_goals=None
    )



def train(context, parameter_path):
    
    course = context.course
    num_courses = 5000

    if course >= num_courses:
        logging.info("Experiment already completed")
        return
    
    logging.info("Resuming the experiment")

    model = HUMN(context.cortex_models, context.hippocampus_models, context.abstraction_models)
    
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


    def near_round(x, base=0.5):
        return np.round(x / base) * base


    def states_to_goals(states, keepdims=True):
        vx = states[:, 5]
        vz = np.abs(states[:, 6])
        sub_states = np.stack([vx, vz], axis=1)
        return sub_states


    def filter(last_pivots, last_scores, stats):
        # accept or reject

        if bool(stats) is False:
            stats["health_max"] = 0
            stats["best_match_distance"] = np.ones([len(context.goals), 1]) * 1e6
            stats["best_match"] = np.zeros([len(context.goals), 2])

        last_goals = last_pivots
        health = np.max(last_scores, axis=0)

        # compute l2 distance to goals
        # goal_array has shape [m, 2]
        # last_goals has shape[n, 2]
        # distance should have shape [m, n]
        distance = np.linalg.norm(np.expand_dims(goal_array, axis=1) - np.expand_dims(last_goals, axis=0), axis=2)
        
        # pick the best for each goal
        best_match_index = np.argmin(distance, axis=1)
        best_match_distance = np.min(distance, axis=1, keepdims=True)
        best_match = last_goals[best_match_index, :]

        survive = health >= 0.5 * stats["health_max"]
        improve_ratio = best_match_distance / (stats["best_match_distance"] + 1e-6)

        stats["health_max"] = np.maximum(stats["health_max"], np.max(health))
        stats["best_match_distance"] = np.minimum(stats["best_match_distance"], best_match_distance)
        stats["best_match"] = np.where(improve_ratio < 1.0, best_match, stats["best_match"])

        if np.any(improve_ratio < 2.0) and np.any(survive):
            return True, stats
        
        return True, stats


    def prepare_data_tuples(premature_termination, states, actions, rewards, num_layers, skip_steps):
        states = np.stack(states, axis=0)
        actions = np.stack(actions, axis=0)
        rewards = np.reshape(np.stack(rewards, axis=0), [-1, 1])
        goals = states_to_goals(states, keepdims=True)

        path_layer_tuples = [] # List[Tuple[algebraic.State_Action_Sequence, algebraic.Pointer_Sequence, algebraic.Expectation_Sequence]]
        for layer_i in range(num_layers - 1):
            path = alg.State_Action_Sequence(states, actions)

            skip_sequence = [i for i in range(0, len(states), skip_steps)]
            # always add the last index
            if skip_sequence[-1] != len(states) - 1:
                skip_sequence.append(len(states) - 1)
            skip_pointer_sequence = alg.Pointer_Sequence(skip_sequence)

            expectation_sequence = alg.Expectation_Sequence(rewards[skip_sequence, :], states[skip_sequence, :])

            path_layer_tuples.append((path, skip_pointer_sequence, expectation_sequence))

            states = states[skip_sequence]
            actions = actions[skip_sequence]
            # reward is the average of the rewards in the skip sequence
            rewards = compute_sum_along_sequence(rewards, skip_sequence) / skip_steps
            goals = compute_average_along_sequence(goals, skip_sequence)

        # final layer
        path = alg.State_Action_Sequence(states, actions)
        skip_pointer_sequence = alg.Pointer_Sequence([i for i in range(0, len(states))])

        discount_kernel = generate_mean_geometric_matrix(states.shape[0], diminishing_factor=0.98, upper_triangle=True)
        discounted_rewards = np.matmul(discount_kernel, rewards)
        discounted_goals = np.matmul(discount_kernel, goals)
        goals = near_round(discounted_goals)
        expectation_sequence = alg.Expectation_Sequence(discounted_rewards, goals)

        path_layer_tuples.append((path, skip_pointer_sequence, expectation_sequence))

        return path_layer_tuples, goals, rewards


    logging.info(f"Training experiment {context.name}")

    env = gym.make(
        'Hopper-v5',
        healthy_reward=1, forward_reward_weight=0, ctrl_cost_weight=1e-3, 
        healthy_angle_range=(-math.pi / 2, math.pi / 2), healthy_state_range=(-100, 100), 
        render_mode=None
    )

    num_layers = len(context.cortex_models)
    random_seed = context.random_seed
    goal_array = np.asarray([goal[0] for goal in context.goals], dtype=np.float32)
    if context.best_goals is not None:
        best_targets = np.array(context.best_goals, dtype=np.float32)
    else:
        best_targets = goal_array

    total_trials = 0
    while course < num_courses:
        logging.info(f"Course {course}")
        random.seed(random_seed)
        
        total_steps = 0
        max_total_steps = 1200
        course_statistics = {}

        epsilon = 0.1

        num_trials = 0
        stamp = time.time()
        while total_steps < max_total_steps:
            
            if course > 0:
                target = best_targets[total_trials % len(context.goals), :]
                stable_state = alg.Expectation(target)
            
            random.seed(random_seed)
            env.action_space.seed(random_seed)
            observation, info = env.reset(seed=random_seed)
            random_seed = random.randint(0, 100000)
            
            states = []
            actions = []
            rewards = []
            premature_termination = False
            for _ in range(400):
                if random.random() <= epsilon or course == 0:
                    selected_action = env.action_space.sample()
                    # quantize
                    selected_action = np.round(selected_action)
                else:
                    a = model.react(alg.State(observation.data), stable_state)
                    selected_action = a.data
                    if selected_action.size < 3:
                        logging.warning(f"Invalid action: {selected_action}")
                        break
                    selected_action += np.random.normal(0, epsilon, size=selected_action.shape)
                    selected_action = np.clip(selected_action, -1, 1)

                next_observation, reward, terminated, truncated, info = env.step(selected_action)

                # check for nan
                if np.isnan(next_observation.data).any() or np.isnan(reward.data).any():
                    logging.warning("Nan detected in observation")
                    break

                states.append(observation)
                actions.append(selected_action)
                rewards.append(reward)
                if terminated or truncated:
                    premature_termination = True
                    break
                else:
                    observation = next_observation

            # now make hierarchical data
            path_layer_tuples, last_pivots, last_scores = prepare_data_tuples(premature_termination, states, actions, rewards, num_layers, context.skip_steps)
            accept_this, course_statistics = filter(last_pivots, last_scores, course_statistics)
            
            if accept_this:
                trainers = model.observe(path_layer_tuples)
                num_trials += 1
                total_trials += 1

                # before_percent = round(total_steps * 100 / max_total_steps)
                total_steps += len(states)
                # after_percent = round(total_steps * 100 / max_total_steps)
                # if before_percent != after_percent:
                #     # print at every 1 % progress
                #     logging.info(f"Env collected: {after_percent:.2f}% (est finish time: {((time.time() - stamp) * (max_total_steps - total_steps) / total_steps):.2f}s)")
                    
        logging.info(f"Average steps: {total_steps/num_trials}")

        for trainer in trainers:
            trainer.prepare_batch(max_mini_batch_size=32, max_learning_sequence=64)

        loop_train(trainers, 100)

        for trainer in trainers:
            trainer.clear_batch()

        course += 1

        context.course = course
        context.random_seed = random_seed
        context.best_goals = course_statistics["best_match"].tolist()
        
        if course % 100 == 0 or course == num_courses:
            Context.save(context, parameter_path)

    env.close()

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
    experiment_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "experiments", "hopper")

    if args.clear:
        # just clear and return
        empty_directory(experiment_path)
        exit(0)

    with experiment_session(experiment_path) as context:
        
        env = gym.make(
            'Hopper-v5',
            healthy_reward=1, forward_reward_weight=0, ctrl_cost_weight=1e-3, 
            healthy_angle_range=(-math.pi, math.pi), healthy_state_range=(-100, 100), 
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
                    if selected_action.size < 3:
                        logging.warning(f"Invalid action: {selected_action}")
                        break
                    observation, reward, terminated, truncated, info = env.step(selected_action)
                    img = env.render()
                    imgs.append(img)
                    if terminated or truncated:
                        break
                logging.info(f"Trial {j} completed with length {len(imgs)}")
                write_gif(imgs, output_gif, fps=30)

        for i, (goal, goal_text) in enumerate(context.goals):
            goal = context.best_goals[i]
            result_path = os.path.join(experiment_path, "results", goal_text)

            model = HUMN(context.cortex_models, context.hippocampus_models, context.abstraction_models)
            observation, info = env.reset()

            # total_steps = 0
            # num_trials = 100
            # print_steps = max(1, num_trials // 10)
            # for j in range(num_trials):
            #     if j % print_steps == 0 and j > 0:
            #         # print at every 1 % progress
            #         logging.info(f"Environment testing: {(j * 100 / num_trials):.2f}")
            #     observation, info = env.reset()
            #     for _ in range(1000):
            #         # selected_action = env.action_space.sample()
            #         a = model.react(alg.State(observation.data), alg.Expectation(goal))
            #         selected_action = np.clip(np.asarray(a.data), -1, 1)
            #         observation, reward, terminated, truncated, info = env.step(selected_action)
            #         total_steps += 1
            #         if terminated or truncated:
            #             break

            # logging.info(f"Parameter set {context.name}; average behavior steps: {total_steps/num_trials}")
            logging.info(f"Parameter set {context.name}; goal: {goal_text}")
                         
            render_path = os.path.join(result_path, "render")
            os.makedirs(render_path, exist_ok=True)

            def generation_action(observation):
                a = model.react(alg.State(observation.data), alg.Expectation(goal))
                return np.clip(np.asarray(a.data), -1, 1)
            
            generate_visual(render_path, 4, generation_action)

        env.close()
