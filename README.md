# Thinking model
An implementation of thinking model version XL (codename: knapsack).

## Knapsack

Using fix object-feature representation to implicit generalization to by-pass the requirement for negative examples.


## Run
* install [docker-ce](https://www.linode.com/docs/guides/installing-and-using-docker-on-ubuntu-and-debian/) 
* adding user to docker group `sudo usermod -aG docker $USER`
* `./build_docker.sh`
* `./run_docker.sh` to open a shell

## To do
* ~Implement knapsack GPU~
* ~Implement trainer~
* ~Test hippocampus new functions (compute_entropy, get_next)~
* Invertible mutual knapsack
* Entropy should be computed from the heuristic model not the hippocampus?
* Clear weight on retrain
