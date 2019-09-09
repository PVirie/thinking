import argparse
import os
import sys
import shutil
import yaml
import utilities as util

import model
import world

dir_path = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(dir_path, "..", "configurations")

parser = argparse.ArgumentParser()
parser.add_argument('--resume', action="store_true")
opts = parser.parse_args()


def get_config(config):
    with open(os.path.join(config_path, config), 'r') as stream:
        return yaml.load(stream)


if __name__ == '__main__':
    config = get_config("prototype.yaml")
