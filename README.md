# Thinking model

An implementation of thinking model version XXXIX(II).

## Discretum

Use hippocampus neighboring and superpositional sum to bypass the requirement for negative examples.

## Prerequisites

-   install [docker-ce](https://www.linode.com/docs/guides/installing-and-using-docker-on-ubuntu-and-debian/)
-   (Optional) install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#getting-started)
-   adding user to docker group `sudo usermod -aG docker $USER`

### Run experiments

-   `./run_manual.sh` The program will automatically random graph and train the parameter. If you want to retrain the parameter, you can run simply delete the weight direction in `./artifacts`.
-   For VSCode, launch `Docker: Python - GPU` configuration. (Docker: Python - NOGPU) for no gpu environment.

## To do

-   World update prior vs bootstrapping (implement statistical keeping of the variance and mean to estimate extreme cases.)
-   Convert all np array to jax array.
-   Test the algorithm with many elaborate example environments.
-   ~Jax~
-   ~Vector field representation~
-   ~Basis learning~
-   ~Test: one step find path without enhancement~
-   ~Cannot be fixed (need repeatative prevention mechanism): Fix cortex loop (caused by unseen pair on estimation.)~
-   ~Proper hierarchy debug~
-   ~Rewrite to next hierarchy function~
-   ~Fix cortex does not obey neighbor~
-   ~Fix hippocampus gap with chunks~
-   ~Fix why cognitive planner is not the min of either hippocampus or cortex.~
-   ~Inner product divergence model.~
-   ~To next hierarchy function~ (removed enhancer to allow generalization.)
-   ~Use discrete logic to cope with signal degradation.~ (Signal enhancement via external input feedback loop.)
-   ~Solve signal degradation by adding hippocampus enhancer~
-   ~Cortex pathway~
-   ~Hippocampus pathway~
-   ~Compute proper cortex prop~ (because using divergence network and directly compute probability.)
-   ~Isolate diminising parameter~
-   ~Entropy should be computed from the heuristic model not the hippocampus? (No)~
-   ~Clear weight on retrain~
