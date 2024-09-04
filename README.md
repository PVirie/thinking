# Thinking model

An implementation of heuristically uncertainty minimization along networks (HUMN).

| Version | Model                                   | Description                                                                                          |
| ------- | --------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| 44      | **Hippocampal Augmented**               | Use hippocampus for tagging                                                                          |
| 43      | Recursive sub-action inference          | Infer sub-actions instead of states                                                                  |
| 42      | Guided Non-deterministic Turing Machine | Use hippocampus neighboring and superpositional sum to bypass the requirement for negative examples. |
| 41      | Freewill                                | Control actions                                                                                      |
| 39(2)   | Discretum                               | Hippocampus + heuristic                                                                              |
| 40      | Knapsack                                | Use knapsack generalization, th basic idea is the concept of property.                               |
| 38      | Subprobability                          | argmax x given s where x is a neighbor of s                                                          |
| 37      | Functional variational                  | Generalize neighbor distribution                                                                     |
| 36      | Gaussian variational                    | Gaussian neighbor distribution                                                                       |
| 35      | The fundamental model                   | Hierarchical Uncertainty Minimization across Network                                                 |

## Prerequisites

-   install [docker-ce](https://www.linode.com/docs/guides/installing-and-using-docker-on-ubuntu-and-debian/)
-   (Optional) install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#getting-started)
-   adding user to docker group `sudo usermod -aG docker $USER`

## Run experiments

-   `./run_manual.sh {path to file}` The program will automatically random graph and train the parameter. If you want to retrain the parameter, you can run simply delete the weight direction in `./artifacts`.
-   For VSCode,
    -   launch `Python - Current file clear` configuration to clear weights.
    -   launch `Python - Current file in gpu docker` configuration.
    -   launch `Python - Current file in cpu docker` for no gpu environment.
-   Running on windows
    -   The relative path in windows that passes to docker has invalid path separator. Using POSIX path separator when passing `{path to file}` parameter when running `run_manual.sh` script. Or simply create a new configuration in `.vscode/launch.json` that fixed the file you want to run with the POSIX path separator.

## To do

-   [x] Hierarchical goal pursuing
    -   [x] Table kernel
    -   [x] Linear kernel
    -   [x] Entropic abstraction
    -   [x] Parallelize layers
    -   [x] Use asymmetric gradient update to keep track of the best so far
-   [x] Enhancement
    -   [x] jax enhance: use jit (remove if else in training functions)
    -   [x] Use optax
    -   [x] Transformer kernel
-   [x] Abstraction
    -   [x] Implement entropy abstraction
        -   [x] Conclusion: entropy might actually increase the steps.
        -   [ ] Hypothesis: does entropy reduce model complexity to learn path?
-   [ ] Hippocampus
    -   [ ] Make core models accept context of shape (batch, context length, feature length)
    -   [ ] added different encodings to the model
-   [ ] Language model experiment (abstraction with embedding)
    -   [ ] Predicting finite token is easier than predicting continuous value
-   [ ] Cart pole experiment
-   [ ] Calculator experiment
-   [ ] Paper topic
    -   [ ] Versioning learning with asymmetric update
