import os
import json
from . import base, table, linear, linear_stat
import inspect


current_path = None

def initialize(path):
    global current_path
    os.makedirs(path, exist_ok=True)
    current_path = path


def recursive_load_model(class_params, model_class):
    # iterate over init arguments
    for arg in inspect.signature(model_class.__init__).parameters:
        # if the arg is a subclass of base.Persistent_Model
        if issubclass(model_class.__init__.__annotations__.get(arg, object), base.Persistent_Model):
            class_params[arg] = load(class_params[arg])
    # remove the class_type and class_name
    class_params.pop("class_type")
    class_params.pop("class_name")
    return model_class(**class_params)


def load(model_id: str):
    with open(os.path.join(current_path, model_id, "class_params.json"), "r") as f:
        class_params = json.load(f)
    class_type = class_params["class_type"]
    class_name = class_params["class_name"]
    if class_type == "model":
        if class_name == "table":
            return recursive_load_model(class_params, table.Model)
        elif class_name == "linear":
            return recursive_load_model(class_params, linear.Model)
    elif class_type == "stat":
        if class_name == "linear":
            return recursive_load_model(class_params, linear_stat.Model)
    return None


def save(model):
    if not model.is_updated:
        return model.instance_id
    model_path = os.path.join(current_path, model.instance_id)
    os.makedirs(model_path, exist_ok=True)
    with open(os.path.join(model_path, "class_params.json"), "w") as f:
        json.dump(model.get_class_parameters(), f)
    model.save(model_path)
    return model.instance_id