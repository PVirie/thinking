#!/bin/bash

# set cwd to current file path
cd "$(dirname "$0")"

# get run configuration from the first argument
profile=$1

# shutdown all running containers
docker compose -f docker_compose.yaml --profile $profile down

# check for flags
if [ -z "$3" ]
then
    docker compose -f docker_compose.yaml --profile $profile run -d --build --service-ports "$profile-service" python3 $2
else
    docker compose -f docker_compose.yaml --profile $profile run -d --build --service-ports "$profile-service" python3 $2 $3
fi

sleep 5