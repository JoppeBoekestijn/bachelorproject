from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# matplotlib inline
import matplotlib as mpl

mpl.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.image import imread
import tensorflow as tf
import numpy as np
import sys
import os

import knifey
from knifey import img_size, img_size_flat, img_shape, num_classes, num_channels
