import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)

from node import Node, Node_tensor_2D, set_node_dim