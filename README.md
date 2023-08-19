# Thinking model

An implementation of thinking model version XXXIX(II).

## Discretum

Use hippocampus neighboring and superpositional sum to bypass the requirement for negative examples.

## Prerequisites

-   install [docker-ce](https://www.linode.com/docs/guides/installing-and-using-docker-on-ubuntu-and-debian/)
-   install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#getting-started)
-   adding user to docker group `sudo usermod -aG docker $USER`
-   `./build_docker.sh`

### Run experiments

-   `./run_docker.sh` to open a shell
    -   `python3 src/strategy.py` for a single run experiment
    -   `python3 tasks/exp_onehot.py` for batch experiments

### Run web view

-   `python3 main.py` to start a web server
    -   `http://localhost:8000/view/index.html`

## To do

-   Use invertible embedding might help generalization?
-   ~Jax
-   ~Vector field representation
-   Test the algorithm with many elaborate example environments.
-   ~Basis learning
-   ~Test: one step find path without enhancement
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
