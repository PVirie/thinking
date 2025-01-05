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

If you want to purely install the python code, you can follow the steps in the docker files.

## Run experiments

-   By default, use program script `./run_manual.sh {configuration} {path to file} {optional flags}` to execute the python file with the selected configuration. (See table below.)
-   For VSCode, press `F5` to run the selected configuration:
    -   launch `jax-cpu clear` for jax in cpu with `--clear` flag.
    -   launch `jax-cpu` for jax in cpu environment.
    -   launch `jax-gpu` for jax in gpu environment.
    -   launch `torch-cpu` for torch in cpu environment
    -   launch `torch-gpu` for torch in gpu environment.
-   Running on Windows
    -   The relative path in Windows that passes to docker has invalid path separators. _Always use POSIX path separators_ when passing `{path to file}` parameter when running `run_manual.sh` script. Or simply create a new configuration in `.vscode/launch.json` with the hard coded configuration you wish to run with the POSIX path separators.

| Experiment         | Task               | Description                                                                            | Valid configurations (pick one)                | File (--flags)                  | Required env vars            |
| ------------------ | ------------------ | -------------------------------------------------------------------------------------- | ---------------------------------------------- | ------------------------------- | ---------------------------- |
| **Benchmark**      | Benchmark devices  | Run the benchmark to compare the performance of the devices.                           | `jax-gpu`, `jax-cpu`, `torch-cpu`, `torch-gpu` | `tasks/benchmark.py`            | -                            |
| **Simple graph**   | Train and test     | Train the model to learn simple graph tasks.                                           | `jax-gpu`, `jax-cpu`                           | `tasks/simple.py`               | -                            |
|                    | Clear weight       | Clear the weight in the model. (Or simply delete the weight direction in `./artifacts` | `jax-gpu`, `jax-cpu`                           | `tasks/simple.py --clear`       | -                            |
| **RL: cart pole**  | Train and test     | Train the model to learn to control the cart pole.                                     | `jax-gpu`, `jax-cpu`                           | `tasks/rl_cart_pole.py`         | -                            |
|                    | Clear weight       | Clear the weight in the model. (Or simply delete the weight direction in `./artifacts` | `jax-gpu`, `jax-cpu`                           | `tasks/rl_cart_pole.py --clear` | -                            |
| **Language model** | Prepare            | Prepare data for the the language model hierarchical guide model.                      | `torch-cpu`                                    | `tasks/lm_data_prepare.py`      | `HF_TOKEN`, `OPENAI_API_KEY` |
|                    | Train hierarchy    | Train the language model hierarchical guide model.                                     | `jax-gpu`, `jax-cpu`                           | `tasks/lm_guide_train.py`       | -                            |
|                    | Generate hierarchy | Generate the language model hierarchical guide model.                                  | `jax-gpu`, `jax-cpu`                           | `tasks/lm_guide_inference.py`   | -                            |
|                    | Interpret          | Given the hierarchy guide, print out the text generation.                              | `torch-cpu`                                    | `tasks/lm_data_interpret`       | `HF_TOKEN`                   |

## To do

### Experiments

-   [x] Hierarchical goal pursuing
    -   [x] Table kernel
    -   [x] Linear kernel
    -   [x] Skip step abstraction
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
    -   [x] Implement K-mean tokenization
    -   [x] Train the LM hierarchical guide model
    -   [ ] Interpretability
    -   [ ] Unix command optimize data
-   [x] Reinforcement learning
    -   [x] [Cart pole] (https://gymnasium.farama.org/environments/classic_control/cart_pole/)
        -   [x] Use average state value as the goal
    -   [x] Use curriculum learning
    -   [ ] [Hopper] (https://gymnasium.farama.org/environments/mujoco/hopper/)
        -   [ ] Provide Skeleton dataset for help exploration
        -   [ ] Goals transition
    -   [ ] [BankHeist] (https://ale.farama.org/environments/bank_heist/)
-   [x] Abstraction
    -   [x] Abstraction interface
    -   [x] Implement ideal entropy abstraction
    -   [ ] Implement entropy abstraction via neural statistic keeping
        -   [ ] Reverse masks (from pivots to states)
-   [x] Hippocampus
    -   [x] Hippocampus interface
    -   [ ] Test positional encoding
    -   [ ] Rotary positional encoding
    -   [ ] Location encoding
    -   [ ] Tagging

### Code

-   [x] Interruptible training
-   [x] Torch GPU
-   [ ] Use flax nnx
