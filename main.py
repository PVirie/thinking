import argparse
import os
import yaml
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn


dir_path = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(dir_path, "configurations")

parser = argparse.ArgumentParser()
parser.add_argument('--resume', action="store_true")
opts = parser.parse_args()


def get_config(config):
    with open(os.path.join(config_path, config), 'r') as stream:
        return yaml.load(stream)


app = FastAPI()

app.mount("/view", StaticFiles(directory="view"), name="web")
app.mount("/artifacts", StaticFiles(directory="artifacts"), name="artifacts")


if __name__ == '__main__':
    config = get_config("prototype.yaml")
    print(config)
    uvicorn.run("main:app", port=8000, log_level="info")
