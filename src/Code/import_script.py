import torch
import torch.nn as nn
import numpy as np
import random
import pickle
import time
import math
import sys
import itertools
import copy
import os
import gpytorch
import gpytorch.constraints
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from collections import defaultdict