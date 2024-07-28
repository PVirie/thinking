import os
import json
from . import base, table, linear, linear_stat



def load(path: str) -> None:
    with open(os.path.join(path, "metadata.json"), "r") as f:
        metadata = json.load(f)
    class_type = metadata["class_type"]
    class_name = metadata["class_name"]
    if class_type == "model":
        if class_name == "table":
            return table.Model.load(path, metadata)
        elif class_name == "linear":
            return linear.Model.load(path, metadata)
    elif class_type == "stat_model":
        if class_name == "linear":
            return linear_stat.Model.load(path, metadata)
    return None


def save(path: str, model) -> None:
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "metadata.json"), "w") as f:
        json.dump(model.get_class_parameters(), f)
    model.save(path)