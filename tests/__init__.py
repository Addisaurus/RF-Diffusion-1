import os
# Set OpenMP environment variable before any other imports
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

# Now it's safe to import other modules
import torch
import numpy as np