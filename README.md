# Thinking model
An implementation of thinking model version XXXIX(II).

## Discretum

Use hippocampus neighboring and superpositional sum to bypass the requirement for negative examples.


## Run
* install [docker-ce](https://www.linode.com/docs/guides/installing-and-using-docker-on-ubuntu-and-debian/) 
* install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#getting-started)
* adding user to docker group `sudo usermod -aG docker $USER`
* `./build_docker.sh`
* `./run_docker.sh` to open a shell

## To do
* Fix cortex loop
* Entropy should be normalized against repetition with bases.
* Basis learning
* Test: one step find path
* ~Inner product divergence model.~
* ~To next hierarchy function~ (removed enhancer to allow generalization.)
* ~Use discrete logic to cope with signal degradation.~ (Signal enhancement via external input feedback loop.)
* ~Solve signal degradation by adding hippocampus enhancer~
* ~Cortex pathway~
* ~Hippocampus pathway~
* ~Compute proper cortex prop~ (because using divergence network and directly compute probability.)
* ~Isolate diminising parameter~
* ~Entropy should be computed from the heuristic model not the hippocampus? (No)~
* ~Clear weight on retrain~

