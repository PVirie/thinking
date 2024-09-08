#!/bin/bash

# first set cwd to current file path
cd "$(dirname "$0")"
# second get run configuration from argument 2, if not set, use default jax-gpu
if [ -z "$2" ]
then
    profile="jax-gpu"
else
    profile=$2
fi
# docker compose -f docker_compose.yaml --profile jax-gpu down
# docker compose -f docker_compose.yaml --profile jax-gpu run -d --build --service-ports jax-gpu-service python3 $1

docker compose -f docker_compose.yaml --profile $profile down
docker compose -f docker_compose.yaml --profile $profile run -d --build --service-ports "$profile-service" python3 $1