# Thinking model

An implementation of Hierarchical Unified Model for Actionable INterpretation.

| Version | Model                                   | Description                                                                                          |
| ------- | --------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| 44      | **Hippocampal Augmented**               | Use hippocampus for tagging                                                                          |
| 43      | Recursive sub-action inference          | Infer sub-actions instead of states                                                                  |
| 42      | Guided Non-deterministic Turing Machine | Use hippocampus neighboring and superpositional sum to bypass the requirement for negative examples. |

## Prerequisites

I have all OSs (Linux, Windows, Mac) in my possession, so I will try to make it work on all of them as far as I can.
But I would recommend using Linux for the best experience.
If you have any issues, please let me know.

-   install docker
    -   Linux, please follow [docker-ce](https://www.linode.com/docs/guides/installing-and-using-docker-on-ubuntu-and-debian/)
    -   Linux, also add your user to docker group `sudo usermod -aG docker $USER`
    -   Windows and Mac, please install [Docker Desktop](https://www.docker.com/products/docker-desktop)
-   gpu support (Optional)
    -   Nvidia driver version 555.xx or higher (for CUDA 12.5.1+)
    -   Linux, install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
    -   Windows, follow [this guide](https://docs.docker.com/desktop/gpu/) to enable gpu support in docker desktop.
    -   Mac, no support yet, but Mac cpu silicons are already great.
-   create `secrets.env` file to install neccessary tokens (Huggingface, OpenAI, etc.) (See the running section for more details.)
    ```
    export VARIABLE_NAME_1="VARIABLE_VALUE"
    export VARIABLE_NAME_2="VARIABLE_VALUE"
    ...
    ```

## Run experiments

-   `./run_manual.sh {path to file} {configuration}` The program will execute the python file with the selected configuration.
-   For VSCode,
    -   launch `Jax - Current file clear` configuration to clear weights depending on how it handle in the file you want to run.
    -   launch `Jax - Current file in gpu docker` for jax in gpu environment.
    -   launch `Jax - Current file in cpu docker` for jax in cpu environment.
    -   launch `Pytorch - Current file in cpu docker` for torch in cpu environment
-   Running on windows
    -   The relative path in windows that passes to docker has invalid path separators. Using POSIX path separator when passing `{path to file}` parameter when running `run_manual.sh` script. Or simply create a new configuration in `.vscode/launch.json` that fixed the file you want to run with the POSIX path separator.

| Experiment         | Task               | Description                                                                            | File                          | Valid configs        | Required env vars            |
| ------------------ | ------------------ | -------------------------------------------------------------------------------------- | ----------------------------- | -------------------- | ---------------------------- |
| **Simple graph**   | Train and test     | Train the model to learn simple graph tasks.                                           | `tasks/simple.py`             | `jax-gpu`, `jax-cpu` | -                            |
|                    | Clear weight       | Clear the weight in the model. (Or simply delete the weight direction in `./artifacts` |                               | `jax-gpu`, `jax-cpu` | -                            |
| **RL: cart pole**  | Train and test     | Train the model to learn to control the cart pole.                                     | `tasks/rl_cart_pole.py`       | `jax-gpu`, `jax-cpu` | -                            |
|                    | Clear weight       | Clear the weight in the model. (Or simply delete the weight direction in `./artifacts` |                               | `jax-gpu`, `jax-cpu` | -                            |
| **Language model** | Prepare            | Prepare data for the the language model hierarchical guide model.                      | `tasks/lm_data_prepare.py`    | `torch-cpu`          | `HF_TOKEN`, `OPENAI_API_KEY` |
|                    | Train hierarchy    | Train the language model hierarchical guide model.                                     | `tasks/lm_guide_train.py`     | `jax-gpu`, `jax-cpu` | -                            |
|                    | Generate hierarchy | Generate the language model hierarchical guide model.                                  | `tasks/lm_guide_inference.py` | `jax-gpu`, `jax-cpu` | -                            |
|                    | Interpret          | Given the hierarchy guide, print out the text generation.                              | `tasks/lm_data_interpret`     | `torch-cpu`          | `HF_TOKEN`                   |

## To do

-   [x] Hierarchical goal pursuing
    -   [x] Table kernel
    -   [x] Linear kernel
    -   [x] Entropic abstraction
    -   [x] Parallelize layers
    -   [x] Use asymmetric gradient update to keep track of the best so far
    -   [x] Reset context after each goal change
-   [x] Enhancement
    -   [x] JIT everything
    -   [x] Use optax
    -   [x] Transformer kernel
    -   [x] Value access vs score access hyperparameter to select which type of hypothesis learning to use.
-   [x] Language model experiment (abstraction with embedding)
    -   [x] Implement torch docker for lowest language model layer and use Think mode for higher layers
    -   [x] Linear embedding transformation kernel
    -   [x] Make core models accept context of shape (batch, context length, feature length)
    -   [x] LLM Steering
    -   [ ] Train the LM hierarchical guide model (in progress, block by the amount of resource to train in my workstation)
-   [x] Reinforcement learning
    -   [x] [Cart pole] (https://gymnasium.farama.org/environments/classic_control/cart_pole/)
        -   [ ] Use average state value as the goal
    -   [ ] [Hopper] (https://gymnasium.farama.org/environments/mujoco/hopper/)
-   [x] Abstraction
    -   [x] Implement entropy abstraction
    -   [ ] Implement Neural statistic keeping
    -   [ ] Hypothesis: does entropy reduce model complexity to learn path?
    -   [ ] Investigate why in think mode, entropic abstraction outperforms the skip abstraction, and opposite results in react mode.
-   [ ] Hippocampus
    -   [ ] Tree based position encoding
-   [ ] Paper content
    -   [ ] Multi-hierarchy model for goal pursuing
        -   [ ] Discount steps
        -   [ ] Abstraction
        -   [ ] Reaction vs Thinking mode
        -   [ ] Algebraic structure
    -   [ ] Asymmetric update
        -   [ ] Value and score access
        -   [ ] Top layer should use value access when we do not know the range of score. But value access should perform better when we want to group action by similarity.
