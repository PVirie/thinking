docker run --name thinking_container --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -u $(id -u):$(id -g) -v $(pwd):/workspace/ -w /workspace -it --rm thinking_image bash
