# Thinking model

An implementation of thinking models.

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

-   `./run_manual.sh` The program will automatically random graph and train the parameter. If you want to retrain the parameter, you can run simply delete the weight direction in `./artifacts`.
-   For VSCode, launch `Docker: Python - GPU` configuration. (Docker: Python - NOGPU) for no gpu environment.

## To do

-   [ ] Implement heuristically uncertainty minimization along networks (HUMN).
    -   [ ] Hierarchical goal pursuing
        -   [x] Table data
        -   [x] Linear kernel
        -   [x] Entropic abstraction
        -   [ ] Maze experiment
        -   [ ] Transformer kernel
    -   [ ] Experiment: incremental learning
    -   [ ] Experiment: added different positional encodings to the model.
-   [x] jax enhance: use jit (remove if else in training functions)
