# Thinking model

An implementation of heuristically uncertainty minimization along networks (HUMN).

| Version | Model                                   | Description                                                                                          |
| ------- | --------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| 44      | **Hippocampal Augmented**               | Use hippocampus for tagging                                                                          |
| 43      | Recursive sub-action inference          | Infer sub-actions instead of states                                                                  |
| 42      | Guided Non-deterministic Turing Machine | Use hippocampus neighboring and superpositional sum to bypass the requirement for negative examples. |

## Prerequisites

-   install [docker-ce](https://www.linode.com/docs/guides/installing-and-using-docker-on-ubuntu-and-debian/)
-   (Optional) install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#getting-started)
-   adding user to docker group `sudo usermod -aG docker $USER`
-   create `secrets.env` file to install neccessary tokens (Huggingface, OpenAI, etc.) (See the running section for more details.)
    ```
    export VARIABLE_NAME_1="VARIABLE_VALUE"
    export VARIABLE_NAME_2="VARIABLE_VALUE"
    ...
    ```

## Run experiments

### General execution

-   `./run_manual.sh {path to file} {configuration}` The program will execute the python file with the selected configuration.
-   For VSCode,
    -   launch `Jax - Current file clear` configuration to clear weights depending on how it handle in the file you want to run.
    -   launch `Jax - Current file in gpu docker` for jax in gpu environment.
    -   launch `Jax - Current file in cpu docker` for jax in cpu environment.
    -   launch `Pytorch - Current file in cpu docker` for torch in cpu environment
-   Running on windows
    -   The relative path in windows that passes to docker has invalid path separators. Using POSIX path separator when passing `{path to file}` parameter when running `run_manual.sh` script. Or simply create a new configuration in `.vscode/launch.json` that fixed the file you want to run with the POSIX path separator.

| Experiment     | Task            | File                  | Description                                                                            | Valid configs    | Required env vars        |
| -------------- | --------------- | --------------------- | -------------------------------------------------------------------------------------- | ---------------- | ------------------------ |
| Simple graph   | Train and test  | `tasks/simple.py`     | Train the model to learn simple graph tasks.                                           | jax-gpu, jax-cpu | -                        |
|                | Clear weight    |                       | Clear the weight in the model. (Or simply delete the weight direction in `./artifacts` | jax-gpu, jax-cpu | -                        |
| Language model | Prepare         | `tasks/lm_prepare.py` | Prepare data for the the language model hierarchical guide model.                      | torch-cpu        | HF_TOKEN, OPENAI_API_KEY |
|                | Train hierarchy | `tasks/lm_train.py`   | Train the language model hierarchical guide model.                                     | jax-gpu, jax-cpu | -                        |
|                | Interpret       | `tasks/lm_interpret`  | Print out the text generation.                                                         | torch-cpu        | HF_TOKEN                 |

## To do

-   [x] Hierarchical goal pursuing
    -   [x] Table kernel
    -   [x] Linear kernel
    -   [x] Entropic abstraction
    -   [x] Parallelize layers
    -   [x] Use asymmetric gradient update to keep track of the best so far
    -   [ ] Hypothesis: reset context after each goal will reduce the complexity of the model?
-   [x] Enhancement
    -   [x] jax enhance: use jit (remove if else in training functions)
    -   [x] Use optax
    -   [x] Transformer kernel
-   [x] Abstraction
    -   [x] Implement entropy abstraction
        -   [x] Conclusion: entropy might actually increase the steps.
        -   [ ] Hypothesis: does entropy reduce model complexity to learn path?
-   [ ] Language model experiment (abstraction with embedding)
    -   [ ] Implement torch docker for lowest language model layer and use Think mode for higher layers
    -   [ ] Linear embedding transformation kernel
    -   [ ] Reset context after each goal change
    -   [ ] Evaluate LLM vs HUMN augmented LLM
    -   [ ] Hypothesis: predicting finite token is easier than predicting continuous value
-   [ ] Hippocampus
    -   [ ] added different encodings to the model
    -   [ ] Experiment with hippocampus: Tree based position encoding
-   [ ] RL experiment
    -   [ ] Cart pole experiment
-   [ ] Paper topic
    -   [ ] Versioning learning with asymmetric update
