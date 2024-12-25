#!/bin/bash

# run docker
# also add these option like in the compose

    # jax-gpu-service:
    #     profiles: ["jax-gpu"]
    #     build:
    #         dockerfile: ./Dockerfile
    #         context: .
    #     container_name: thinking-gpu
    #     restart: no
    #     volumes:
    #         - ./artifacts/log/:/app/log
    #         - ./artifacts/cache:/app/cache
    #         - ./artifacts/experiments:/app/experiments
    #         - ./humn:/app/humn
    #         - ./core:/app/core
    #         - ./implementations:/app/implementations
    #         - ./tasks:/app/tasks
    #     environment:
    #         - ERROR_LOG=/app/log/error.log
    #         - LOG_LEVEL=info
    #         - PYTHONUNBUFFERED=TRUE
    #         - TF_CPP_MIN_LOG_LEVEL=0
    #         - MUJOCO_GL=osmesa
    #     env_file:
    #         - ./secrets.env
    #     ports:
    #         - "127.0.0.1:5678:5678"
    #     networks:
    #         - app_network
    #     deploy:
    #         resources:
    #             reservations:
    #                 devices:
    #                     - driver: nvidia
    #                       count: 1
    #                       capabilities: [gpu]

cd "$(dirname "$0")"
docker stop thinking-jax-gpu
docker run -d --gpus all \
    -v ./artifacts/log/:/app/log \
    -v ./artifacts/cache:/app/cache \
    -v ./artifacts/experiments:/app/experiments \
    -v ./humn:/app/humn \
    -v ./core:/app/core \
    -v ./implementations:/app/implementations \
    -v ./tasks:/app/tasks \
    -e ERROR_LOG=/app/log/error.log \
    -e LOG_LEVEL=info \
    -e PYTHONUNBUFFERED=TRUE \
    -e TF_CPP_MIN_LOG_LEVEL=0 \
    -e MUJOCO_GL=osmesa \
    --env-file ./secrets.env \
    -p "127.0.0.1:5678:5678" \
    thinking-jax-gpu