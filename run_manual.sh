#!/bin/bash

docker compose -f docker_compose.yaml --profile gpu down
docker compose -f docker_compose.yaml --profile gpu up -d --build --force-recreate