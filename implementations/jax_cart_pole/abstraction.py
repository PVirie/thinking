from humn import abstraction_model, trainer
from typing import Tuple, Union

import jax
import jax.numpy as jnp
import os
import json
import random

try:
    from .algebric import *
except:
    from algebric import *

try:
    import core
except:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
    import core


class Model(abstraction_model.Model):
    pass


if __name__ == "__main__":
    pass
