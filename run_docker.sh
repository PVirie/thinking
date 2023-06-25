#!/bin/bash


# Detect the operating system
OS=$(uname)


# Check if there are any running Docker containers
if [ "$(docker ps -q -a | wc -l)" -gt 0 ]; then
    docker stop $(docker ps -a -q)
fi

docker compose -f docker_compose.yaml down

if [ "$OS" = "Darwin" ]; then
    # MacOS
    docker compose -f docker_compose.yaml --profile no_gpu up -d --build --force-recreate
else
    echo "Using GPU for: $OS"
    docker compose -f docker_compose.yaml --profile with_gpu up -d --build --force-recreate
fi

