#!/bin/bash

# first set cwd to current file path
cd "$(dirname "$0")"
docker compose -f docker_compose.yaml --profile gpu down
docker compose -f docker_compose.yaml --profile gpu run -d --build --service-ports main-gpu python3 $1