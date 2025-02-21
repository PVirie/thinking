import os
import logging
import contextlib
import random
import json
from typing import List, Any
from pydantic import BaseModel
import argparse
import sys
import jax
import jax.numpy as jnp
import gymnax
import gym
# gymnax visualizer is buggy
# from gymnax.visualize import Visualizer

import numpy as np
np.bool8 = np.bool

from utilities.utilities import empty_directory, write_gif

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

    average_random_steps: float = 0
    random_seed: int

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
                context = Context(
                    parameter_sets=parameter_sets, 
                    average_random_steps=metadata.get("average_random_steps", 0), 
                    random_seed=metadata["random_seed"],
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
                "random_seed": self.random_seed,
                "training_state": self.training_state
            }, f, indent=4)
        return True


def setup():
    random_seed = random.randint(0, 1000)
    random.seed(random_seed)

    parameter_sets = []
    ############################# SET 1 ################################

    name = "Kindergarden (Skip steps)"

    state_dim = 4
    action_dim = 1
    reward_dim = 1
    context_length = 1

    cortex_models = [
        cortex.Model(
            0, return_action=True, use_reward=False, use_monte_carlo=True, 
            model=transformer.Model([state_dim, action_dim, state_dim], context_length, 64, [64, 64], memory_size=4, lr=0.0001, r_seed=random_seed),
            trainer=cortex.Trainer_Online()
        ),
        cortex.Model(
            1, return_action=False, use_reward=True, use_monte_carlo=True, 
            model=transformer.Model([state_dim, state_dim, reward_dim], context_length, 64, [64, 64], memory_size=4, lr=0.0001, r_seed=random_seed),
            trainer=cortex.Trainer_Online()
        ),
    ]

    hippocampus_models = [
        hippocampus.Model(state_dim),
        hippocampus.Model(state_dim),
        hippocampus.Model(state_dim)
    ]

    abstraction_models = []

    parameter_sets.append({
        "cortex_models": cortex_models,
        "hippocampus_models": hippocampus_models,
        "abstraction_models": abstraction_models,
        "name": name
    })


    ############################# SET 2 ################################

    name = "Curriculum (Skip steps)"

    state_dim = 4
    action_dim = 1
    expectation_dim = 1
    context_length = 1

    cortex_models = [
        cortex.Model(
            0, return_action=True, use_reward=False, use_monte_carlo=True, 
            model=transformer.Model([state_dim, action_dim, state_dim], context_length, 64, [64, 64], memory_size=4, lr=0.0001, r_seed=random_seed),
            trainer=cortex.Trainer_Online()
        ),
        cortex.Model(
            1, return_action=False, use_reward=True, use_monte_carlo=True, 
            model=transformer.Model([state_dim, state_dim, expectation_dim], context_length, 64, [64, 64], memory_size=4, lr=0.0001, r_seed=random_seed),
            trainer=cortex.Trainer_Online()
        ),
    ]

    hippocampus_models = [
        hippocampus.Model(state_dim),
        hippocampus.Model(state_dim),
        hippocampus.Model(state_dim)
    ]

    abstraction_models = []

    parameter_sets.append({
        "cortex_models": cortex_models,
        "hippocampus_models": hippocampus_models,
        "abstraction_models": abstraction_models,
        "name": name
    })

    return Context(parameter_sets=parameter_sets, random_seed=random_seed)


def compute_sum_along_sequence(x, sequence):
    # input x has shape [sequence, ...]
    # sequence is a list of lengths
    # output has shape [len(sequence), ...], use the delta of cumulative sum to compute the sum from the index to before the next index
    sequence = jnp.array(sequence)
    sequence = jnp.pad(sequence, (1, 0), 'constant', constant_values=0) + 1
    zero_element = jnp.zeros([1, *x.shape[1:]])
    x = jnp.concatenate([zero_element, x, zero_element], axis=0)
    x = jnp.cumsum(x, axis=0)
    x = x[sequence[1:] - 1] - x[sequence[:-1] - 1]
    return x


def prepare_data_tuples(states, actions, rewards, num_layers, skip_steps, input_as_stacks=False):
    if not input_as_stacks:
        states = jnp.stack(states, axis=0)
        actions = jnp.stack(actions, axis=0)
        rewards = jnp.stack(rewards, axis=0)

    actions = jnp.reshape(actions, (-1, 1))
    rewards = jnp.reshape(rewards, (-1, 1))

    path_layer_tuples = [] # List[Tuple[algebraic.State_Action_Sequence, algebraic.Pointer_Sequence, algebraic.Expectation_Sequence]]
    for layer_i in range(num_layers - 1):
        path = alg.State_Action_Sequence(states, actions, rewards)

        skip_sequence = [i for i in range(skip_steps, len(states), skip_steps)]
        # always add the last index
        if len(skip_sequence) == 0 or skip_sequence[-1] != len(states) - 1:
            skip_sequence.append(len(states) - 1)
        skip_pointer_sequence = alg.Pointer_Sequence(skip_sequence)

        expectation_sequence = alg.Expectation_Sequence(states[skip_sequence, :])

        path_layer_tuples.append((path, skip_pointer_sequence, expectation_sequence))

        states = states[skip_sequence, ...]
        actions = actions[skip_sequence, ...]
        # reward is the average of the rewards in the skip sequence
        rewards = compute_sum_along_sequence(rewards, skip_sequence) / skip_steps
    
    # final layer
    path = alg.State_Action_Sequence(states, actions, rewards)
    skip_pointer_sequence = alg.Pointer_Sequence([len(states) - 1])
    expectation_sequence = alg.Expectation_Sequence(jnp.ones((1, 1)))

    path_layer_tuples.append((path, skip_pointer_sequence, expectation_sequence))
    
    return path_layer_tuples


def rollout(r_key, env, env_params, jitted_step, model, total_steps):
    """Rollout a jitted gymnax episode with lax.scan."""
    # Reset the environment
    r_key, key_reset = jax.random.split(r_key)
    obs, state = env.reset(key_reset, env_params)

    stable_state = alg.Expectation([1])
    def react_func(obs):
        return model.react(alg.State(obs), stable_state)

    def policy_step(state_input, tmp):
        """lax.scan compatible step transition in jax env."""
        obs, state, r_key, valid = state_input
        r_key, key_epsilon, key_act, key_step = jax.random.split(r_key, 4)

        random_action = env.action_space(env_params).sample(key_act)
        model_action_obj = react_func(obs)
        model_action = jnp.reshape(jnp.where(model_action_obj.data > 0.5, 1, 0), ())
        epsilon = jax.random.uniform(key_epsilon, shape=random_action.shape)
        selection_action = jnp.where(epsilon < 0.25, random_action, model_action)

        next_obs, next_state, reward, done, _ = jitted_step(key_step, state, selection_action, env_params)
        next_valid = (valid * (1 - done)).astype(jnp.bool_)
        carry = [next_obs, next_state, r_key, next_valid]
        return carry, [obs, selection_action, reward, next_obs, next_valid]

    # Scan over episode step loop
    _, scan_out = jax.lax.scan(
        policy_step,
        [obs, state, r_key, True],
        (),
        total_steps
    )
    # Return masked sum of rewards accumulated by agent in episode
    obs, action, reward, next_obs, done = scan_out
    return obs, action, reward, next_obs, done


def train_model(env, env_params, jitted_step, name, model, num_layers, skip_steps, random_seed, previous_model=None):

    logging.info(f"Training experiment {name}")

    random.seed(random_seed)
    r_key = jax.random.key(random_seed)
    
    stable_state = alg.Expectation([1])

    total_steps = 0
    num_trials = 2000
    print_steps = max(10, num_trials // 100)
    for i in range(num_trials):
        r_key, key_reset = jax.random.split(r_key)
        observation, state = env.reset(key_reset, env_params)

        states = []
        actions = []
        rewards = []
        for _ in range(500):
            if previous_model is None or random.random() < 0.25:
                r_key, key_act = jax.random.split(r_key)
                selected_action = env.action_space(env_params).sample(key_act)
            else:
                a = previous_model.react(alg.State(observation), stable_state)
                # selected_action = 1 if jnp.asarray(a.data)[0].item() > 0.5 else 0
                selected_action = jnp.reshape(jnp.where(a.data > 0.5, 1, 0), ())

            r_key, key_step = jax.random.split(r_key)
            next_observation, state, reward, done, _ = jitted_step(key_step, state, selected_action, env_params)
            states.append(observation)
            actions.append(selected_action)
            rewards.append(reward)
            if done:
                break
            else:
                observation = next_observation
        total_steps += len(states)
        path_layer_tuples = prepare_data_tuples(states, actions, rewards, num_layers, skip_steps, input_as_stacks=False)
        trainers = model.observe(path_layer_tuples)

        # states, actions, rewards, _, dones = rollout(r_key, env, env_params, jitted_step, model, 200)
        # states = states[dones, ...]
        # actions = actions[dones, ...]
        # rewards = rewards[dones, ...]
        # total_steps += states.shape[0]
        # path_layer_tuples = prepare_data_tuples(states, actions, rewards, num_layers, skip_steps, input_as_stacks=True)
        # trainers = model.observe(path_layer_tuples)

        if i % print_steps == 0 and i > 0:
            # print at every 1 % progress
            logging.info(f"Environment collection: {(i * 100 / num_trials):.2f}")
            logging.info(f"Layer loss: {'| '.join([f'{i}, {trainer.get_loss():.4f}' for i, trainer in enumerate(trainers)])}")

    logging.log(logging.INFO, f"Average steps: {total_steps/num_trials}")

    average_random_steps = total_steps/num_trials
    return average_random_steps



def train(context, parameter_path):

    random_seed = context.random_seed
    env, env_params = gymnax.make("CartPole-v1")
    jitted_step = jax.jit(env.step)

    if context.training_state == 0:

        cortex_models = context.parameter_sets[0]["cortex_models"]
        hippocampus_models = context.parameter_sets[0]["hippocampus_models"]
        abstraction_models = context.parameter_sets[0]["abstraction_models"]
        name = context.parameter_sets[0]["name"]

        model = HUMN(cortex_models, hippocampus_models, abstraction_models)

        skip_steps = 8
        num_layers = len(cortex_models)

        average_random_steps = train_model(env, env_params, jitted_step, name, model, num_layers, skip_steps, random_seed)

        previous_model = model

        context.average_random_steps = average_random_steps

        random_seed = random.randint(0, 100000)
        context.random_seed = random_seed
        context.training_state = 1
        Context.save(context, parameter_path)

    else:

        cortex_models = context.parameter_sets[0]["cortex_models"]
        hippocampus_models = context.parameter_sets[0]["hippocampus_models"]
        abstraction_models = context.parameter_sets[0]["abstraction_models"]

        previous_model = HUMN(cortex_models, hippocampus_models, abstraction_models)


    cortex_models = context.parameter_sets[1]["cortex_models"]
    hippocampus_models = context.parameter_sets[1]["hippocampus_models"]
    abstraction_models = context.parameter_sets[1]["abstraction_models"]
    name = context.parameter_sets[1]["name"]

    model = HUMN(cortex_models, hippocampus_models, abstraction_models)

    random_seed = random.randint(0, 100000)
    
    skip_steps = 8
    num_layers = len(cortex_models)

    course = context.training_state - 1

    while course < 1:
        logging.info(f"Course {course}")

        train_model(env, env_params, jitted_step, name, model, num_layers, skip_steps, random_seed, previous_model)

        previous_model = model
        course += 1

        random_seed = random.randint(0, 100000)
        context.random_seed = random_seed
        context.training_state = course + 1
        Context.save(context, parameter_path)

    # env.close()


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
    experiment_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "experiments", "cart_pole")

    if args.clear:
        # just clear and return
        empty_directory(experiment_path)
        exit(0)

    with experiment_session(experiment_path) as context:
        
        env = gym.make("CartPole-v1", render_mode="rgb_array")
        env.action_space.seed(random.randint(0, 1000))

        def generate_visual(render_path, num_trials, action_method):
            for j in range(num_trials):
                observation, info = env.reset(seed=random.randint(0, 1000))
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

        logging.info("----------------- Testing random behavior -----------------")
        logging.info(f"Baseline number of random steps: {context.average_random_steps}")
        # result_path = os.path.join(experiment_path, "results", f"random")
        # render_path = os.path.join(result_path, "render")
        # os.makedirs(render_path, exist_ok=True)
        # generate_visual(render_path, 5, lambda obs: env.action_space.sample())

        for i, parameter_set in enumerate(context.parameter_sets):
            logging.info(f"----------------- Testing model behavior {i} -----------------")
            result_path = os.path.join(experiment_path, "results", f"set_{i}")

            model = HUMN(**parameter_set)
            stable_state = alg.Expectation([1])

            total_steps = 0
            num_trials = 100
            print_steps = max(1, num_trials // 10)
            for j in range(num_trials):
                if j % print_steps == 0 and j > 0:
                    # print at every 10% progress
                    logging.info(f"Environment testing: {(j * 100 / num_trials):.2f}")
                observation, info = env.reset(seed=random.randint(0, 1000))
                for _ in range(500):
                    # selected_action = env.action_space.sample()
                    a = model.react(alg.State(observation.data), stable_state)
                    selected_action = 1 if np.asarray(a.data)[0].item() > 0.5 else 0
                    observation, reward, terminated, truncated, info = env.step(selected_action)
                    total_steps += 1
                    if terminated or truncated:
                        break

            set_name = parameter_set["name"]
            logging.info(f"Parameter set {set_name}; average behavior steps: {total_steps/num_trials}")

            render_path = os.path.join(result_path, "render")
            os.makedirs(render_path, exist_ok=True)

            def generation_action(observation):
                a = model.react(alg.State(observation.data), stable_state)
                return 1 if np.asarray(a.data)[0].item() > 0.5 else 0
            
            generate_visual(render_path, 5, generation_action)

        env.close()