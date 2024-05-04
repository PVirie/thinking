# Thinking model

An implementation of thinking models.

| Version | Model                                   | Description                                                                                          |
| ------- | --------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| 43      | **Recursive sub-action inference**      | Infer sub-actions instead of states                                                                  |
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
-   [ ] Use human on Turing machine
-   [ ] Test the algorithm with many elaborate example environments.
    -   2D maze with 2D position as the state.
-   [ ] jax enhance phase III: use jit (remove if else in training functions)
-   [x] Heuristic learning value estimate using max value from hippocampus
-   [x] Improve metric model learning (added unvisit factor)
-   [x] Fix cortex pathway with ideal model
-   [x] World update prior vs bootstrapping (implement statistical keeping of the variance and mean to estimate extreme cases.)
-   [x] Transformer model
-   [x] jax enhance phase II: reduce data transfer in training (target utilities function)
-   [x] Convert all np array to jax array.
-   [x] Vector field representation
-   [x] Basis learning
-   [x] Test: one step find path without enhancement
-   [x] Cannot be fixed (need repeatative prevention mechanism): Fix cortex loop (caused by unseen pair on estimation.)
-   [x] Proper hierarchy debug
-   [x] Rewrite to next hierarchy function
-   [x] Fix cortex does not obey neighbor
-   [x] Fix hippocampus gap with chunks
-   [x] Fix why cognitive planner is not the min of either hippocampus or cortex.
-   [x] Inner product divergence model.
-   [x] To next hierarchy function (removed enhancer to allow generalization.)
-   [x] Use discrete logic to cope with signal degradation. (Signal enhancement via external input feedback loop.)
-   [x] Solve signal degradation by adding hippocampus enhancer
-   [x] Cortex pathway
-   [x] Hippocampus pathway
-   [x] Compute proper cortex prop (because using divergence network and directly compute probability.)
-   [x] Isolate diminising parameter
-   [x] Entropy should be computed from the heuristic model not the hippocampus? (No)
-   [x] Clear weight on retrain
