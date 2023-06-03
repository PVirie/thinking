#!/bin/bash

# Check if there are any running Docker containers
if [ "$(docker ps -q -a | wc -l)" -gt 0 ]; then
    docker stop $(docker ps -a -q)
fi

docker compose -f docker_compose.yaml down
docker compose -f docker_compose.yaml up -d --build --force-recreate