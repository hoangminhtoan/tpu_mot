import os
import types
from pathlib import Path

config = types.SimpleNamespace()

# get home directory
config.HOME_DIR = str(Path(__file__).parent)

# get pretrained models
config.WEIGHT_DIR = os.path.join(config.HOME_DIR, 'weights')

# get input data videos
config.INPUT_DIR = os.path.join(config.HOME_DIR, 'data')

# get output results
config.RESULT_DIR = os.path.join(config.HOME_DIR, 'results')



