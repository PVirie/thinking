import os
import logging
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
        cortex.Model(
            0, return_action=True, use_reward=False, use_monte_carlo=True, step_discount_factor=0.9,  
            model=transformer.Model([state_dim, action_dim, state_dim], context_length, 128, [256, 256], memory_size=4, value_access=True, lr=0.0001, r_seed=random_seed),
            trainer=cortex.Trainer(total_keeping=200)
        ),
        cortex.Model(
            1, return_action=False, use_reward=True, use_monte_carlo=False, step_discount_factor=0.9, 
            model=transformer.Model([state_dim, state_dim, expectation_dim], context_length, 512, [512, 512], memory_size=16, value_access=True, lr=0.0001, r_seed=random_seed),
            trainer=cortex.Trainer(total_keeping=200)
        )
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
            ([2, 4], "jump forward"),
            ([0, 4], "jump still"),
            ([-2, 4], "jump backward"),
        ],
        random_seed=random_seed,
        course=0,
        best_goals=None
    )


def states_to_goal_statistics(states, keepdims=True):
    vx = states[:, 5]
    vz = np.abs(states[:, 6])
    sub_states = np.stack([vx, vz], axis=1)
    return sub_states


def generate_visual(env, render_path, num_trials, action_method):
    observation, info = env.reset()
    for j in range(num_trials):
        observation, info = env.reset()
        output_gif = os.path.join(render_path, f"trial_{j}.gif")
        imgs = []
        states = []
        for _ in range(500):
            selected_action = action_method(observation)
            if selected_action.size < 3:
                logging.warning(f"Invalid action: {selected_action}")
                break
            observation, reward, terminated, truncated, info = env.step(selected_action)
            img = env.render()
            imgs.append(img)
            states.append(observation)
            if terminated or truncated:
                break
        logging.info(f"Trial {j} completed with length {len(imgs)}")
        states = np.stack(states, axis=0)
        goal_stat = states_to_goal_statistics(states, keepdims=True)
        texts = [f"VX: {goal_stat[i, 0]:.2f}\nVY: {goal_stat[i, 1]:.2f}" for i in range(goal_stat.shape[0])]
        write_gif_with_text(imgs, texts, output_gif, fps=30)


def test(context, parameter_path):
    # RUN TEST 

    env = gym.make(
        'Hopper-v5',
        healthy_reward=0, 
        forward_reward_weight=0, 
        ctrl_cost_weight=0, 
        healthy_angle_range=(-math.pi, math.pi),
        healthy_state_range=(-100, 100),
        healthy_z_range = (0.7, 100.0),
        render_mode="rgb_array", width=1280, height=720
    )
    env.action_space.seed(random.randint(0, 1000))
    observation, info = env.reset(seed=random.randint(0, 1000))

    for i, (goal, goal_text) in enumerate(context.goals):
        # goal = context.best_goals[i]
        result_path = os.path.join(parameter_path, "results", goal_text)

        model = HUMN(context.cortex_models, context.hippocampus_models, context.abstraction_models)
        observation, info = env.reset()

        # logging.info(f"Parameter set {context.name}; average behavior steps: {total_steps/num_trials}")
        logging.info(f"Parameter set {context.name}; goal: {goal_text}")
                        
        render_path = os.path.join(result_path, "render")
        os.makedirs(render_path, exist_ok=True)

        def generation_action(observation):
            a = model.react(alg.State(observation), alg.Expectation(goal))
            return np.clip(np.asarray(a.data), -1, 1)
        
        generate_visual(env, render_path, 4, generation_action)

    env.close()
    

class Experiment_Session:
    def __init__(self, path, force_clear=None):
        self.path = path
        core.initialize(os.path.join(path, "core"))
        if force_clear is not None and force_clear:
            logging.info(f"Clearing the experiment directory: {path}")
            empty_directory(path)
        self.context = Context.load(path)
        if self.context is None:
            self.context = setup()

    def __enter__(self):
        logging.info('Starting session ...')
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        logging.info('Ending session ...')
        if exc_type is not None:
            logging.error(f"Error: {exc_type}, {traceback}")
        else:
            test(self.context, self.path)
        return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    experiment_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "experiments", "hopper")

    if args.clear:
        # just clear and return
        empty_directory(experiment_path)
        exit(0)

    with Experiment_Session(experiment_path) as session:

        context = session.context
        course = context.course
        num_courses = 10

        if course >= num_courses:
            logging.info("Training already completed")
            raise Exception("Experiment already completed")
        
        logging.info("Resuming the experiment")

        model = HUMN(context.cortex_models, context.hippocampus_models, context.abstraction_models)
        
        def loop_train(trainers, num_epoch=1000):
            print_steps = max(10, num_epoch // 100)
            stamp = time.time()
            for i in range(num_epoch):
                for trainer in trainers:
                    trainer.step_update()
                if (i - 1) % print_steps == 0:
                    # print at every 1 % progress
                    # compute time to finish in seconds
                    logging.info(f"Training progress: {(i * 100 / num_epoch):.2f}, time to finish: {((time.time() - stamp) * (num_epoch - i) / i):.2f}s")
                    logging.info(f"Layer loss: {'| '.join([f'{j}, {trainer.get_loss():.4f}' for j, trainer in enumerate(trainers)])}")
            logging.info(f"Learning time {time.time() - stamp}s")


        def update_statistics(last_goals, health, stats):
            # accept or reject

            if bool(stats) is False:
                stats["health_max"] = 0
                stats["best_match_distance"] = np.ones([len(context.goals), 1]) * 1e6
                stats["best_match"] = np.zeros([len(context.goals), 2])

            # compute l2 distance to goals
            # goal_array has shape [m, 2]
            # last_goals has shape[n, 2]
            # distance should have shape [m, a
            distance = np.linalg.norm(np.expand_dims(goal_array, axis=1) - np.expand_dims(last_goals, axis=0), axis=2, ord=1)
            
            # pick the best for each goal
            best_match_index = np.argmin(distance, axis=1)
            best_match_distance = np.min(distance, axis=1, keepdims=True)
            best_match = last_goals[best_match_index, :]

            improve_ratio =  stats["best_match_distance"] / (best_match_distance + 1e-6)

            stats["health_max"] = max(stats["health_max"], health)
            stats["best_match_distance"] = np.minimum(stats["best_match_distance"], best_match_distance)
            stats["best_match"] = np.where(improve_ratio > 1.0, best_match, stats["best_match"])

            return stats
        

        def prepare_data_tuples(premature_termination, target, states, actions, rewards, num_layers, skip_steps):
            states = np.stack(states, axis=0)
            actions = np.stack(actions, axis=0)
            rewards = np.reshape(np.stack(rewards, axis=0), [-1, 1])
            goal_stat = states_to_goal_statistics(states, keepdims=True)
            
            # sign_goal_stat = np.where(abs(goal_stat) < 1e-6, 0, np.sign(goal_stat))
            if target[0] >= 1:
                # forward target
                goal_rewards = goal_stat[:, 0:1] * 0.1
            elif target[0] <= -1:
                # backward target
                goal_rewards = -goal_stat[:, 0:1] * 0.1
            else:
                goal_rewards = -np.abs(goal_stat[:, 0:1]) + 0.1 * goal_stat[:, 1:2]

            # override reward to add control cost
            rewards = goal_rewards + rewards

            path_layer_tuples = [] # List[Tuple[algebraic.State_Action_Sequence, algebraic.Pointer_Sequence, algebraic.Expectation_Sequence]]
            for layer_i in range(num_layers - 1):
                path = alg.State_Action_Sequence(states, actions, rewards)

                skip_sequence = [i for i in range(skip_steps, len(states), skip_steps)]
                # always add the last index
                if len(skip_sequence) == 0 or skip_sequence[-1] != len(states) - 1:
                    skip_sequence.append(len(states) - 1)
                skip_pointer_sequence = alg.Pointer_Sequence(skip_sequence)

                # # pivot is the local maximum hight and local minimum height indices
                # heights = states[:, 0]
                # # smooth heights
                # heights = np.convolve(heights, np.ones(skip_steps) / skip_steps, mode="same")
                # skip_sequence = find_local_extreme_locations(heights)
                # skip_pointer_sequence = alg.Pointer_Sequence(skip_sequence)

                expectation_sequence = alg.Expectation_Sequence(states[skip_sequence, :])

                path_layer_tuples.append((path, skip_pointer_sequence, expectation_sequence))

                states = states[skip_sequence]
                actions = actions[skip_sequence]
                # reward is the sum of the rewards in the skip sequence
                rewards = compute_sum_along_sequence(rewards, skip_sequence) / 100

            # final layer
            path = alg.State_Action_Sequence(states, actions, rewards)
            skip_pointer_sequence = alg.Pointer_Sequence([len(states) - 1])
            expectation_sequence = alg.Expectation_Sequence(np.reshape(target, [1, 2]))
            path_layer_tuples.append((path, skip_pointer_sequence, expectation_sequence))

            goal_stat = np.mean(goal_stat, axis=0, keepdims=True)
            return path_layer_tuples, goal_stat


        logging.info(f"Training experiment {context.name}")

        env = gym.make(
            'Hopper-v5',
            healthy_reward=1, 
            forward_reward_weight=0,
            ctrl_cost_weight=1e-3,
            healthy_angle_range=(-math.pi/2, math.pi/2),
            healthy_state_range=(-100, 100),
            healthy_z_range = (0.7, 100.0),
            render_mode=None
        )

        num_layers = len(context.cortex_models)
        random_seed = context.random_seed
        goal_array = np.asarray([goal[0] for goal in context.goals], dtype=np.float32)

        statistics = {}

        total_trials = 0
        while course < num_courses:
            logging.info(f"Course {course}")
            random.seed(random_seed)
            
            total_steps = 0
            max_total_steps = 200000

            epsilon = 0.3 - 0.2 * (course + 1) / num_courses

            num_trials = 0
            stamp = time.time()
            last_percent_collection = 0
            while True:
                
                selected_target = goal_array[total_trials % len(context.goals), :]
                
                random.seed(random_seed)
                env.action_space.seed(random_seed)
                observation, info = env.reset(seed=random_seed)
                random_seed = random.randint(0, 100000)

                states = []
                actions = []
                rewards = []
                for _ in range(2000):
                    if random.random() <= epsilon or course == 0:
                        selected_action = env.action_space.sample()
                        # quantize
                        selected_action = np.round(selected_action)
                    else:
                        a = model.react(alg.State(observation), alg.Expectation(selected_target))
                        selected_action = a.data
                        if selected_action.size < 3:
                            logging.warning(f"Invalid action: {selected_action}")
                            break
                        selected_action += np.random.normal(0, epsilon, size=selected_action.shape)
                        selected_action = np.clip(selected_action, -1, 1)

                    next_observation, reward, terminated, truncated, info = env.step(selected_action)

                    states.append(observation)
                    actions.append(selected_action)
                    rewards.append(reward)
                    
                    observation = next_observation
                    if terminated:
                        break
                    elif truncated:
                        # exceed the time limit of the env, reset and continue
                        _, info = env.reset(seed=random_seed)
                        random_seed = random.randint(0, 100000)

                # now make hierarchical data
                path_layer_tuples, last_pivots = prepare_data_tuples(terminated, selected_target, states, actions, rewards, num_layers, context.skip_steps)
                statistics = update_statistics(last_pivots, len(states), statistics)
                
                trainers = model.observe(path_layer_tuples)
                num_trials += 1
                total_trials += 1
                total_steps += len(states)
                
                current_percent_collection = round(total_steps * 100 / max_total_steps)
                if current_percent_collection - last_percent_collection >= 10 :
                    # print at every 10% progress
                    logging.info(f"Env collected: {current_percent_collection:.2f}% (est finish time: {((time.time() - stamp) * (max_total_steps - total_steps) / total_steps):.2f}s)")
                    last_percent_collection = current_percent_collection

                if total_steps >= max_total_steps:
                    break

            logging.info(f"Average steps: {total_steps/num_trials}")

            for trainer in trainers:
                trainer.prepare_batch(max_mini_batch_size=32, max_learning_sequence=64)
                
            loop_train(trainers, 1000)

            # for trainer in trainers:
            #     trainer.clear_batch()

            course += 1

            # # disable monte carlo after half of the courses
            # if course == num_courses // 2:
            #     for cortex in context.cortex_models:
            #         cortex.set_update_mode(use_monte_carlo=False)

            context.course = course
            context.random_seed = random_seed
            context.best_goals = statistics["best_match"].tolist()
            
            Context.save(context, experiment_path)
            test(context, experiment_path)

        env.close()
