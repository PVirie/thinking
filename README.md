# Thinking model
An implementation of thinking model version XXXIX(II).

## Discretum

Use hippocampus neighboring and superpositional sum to bypass the requirement for negative examples.


## Run
* install [docker-ce](https://www.linode.com/docs/guides/installing-and-using-docker-on-ubuntu-and-debian/) 
* adding user to docker group `sudo usermod -aG docker $USER`
* `./build_docker.sh`
* `./run_docker.sh` to open a shell

## To do
* ~Solve signal degradation by adding hippocampus enhancer~
* ~Cortex pathway~
* ~Hippocampus pathway~
* ~Compute proper cortex prop~ (because using divergence network and directly compute probability.)
* Entropy should be computed from the heuristic model not the hippocampus?
* ~Clear weight on retrain~

