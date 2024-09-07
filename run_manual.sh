#!/bin/bash

# first set cwd to current file path
cd "$(dirname "$0")"
docker compose -f docker_compose.yaml --profile jax-gpu down
docker compose -f docker_compose.yaml --profile jax-gpu run -d --build --service-ports jax-gpu-service python3 $1