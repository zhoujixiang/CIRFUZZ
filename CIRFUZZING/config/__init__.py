import config.lidar_config
import config.common_config
import config.v2x_config
import os
import sys

modality = "multi"
not_behind_initial_obj = True

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ['PROJECT_DIR'] = _ROOT
sys.path.append(_ROOT)
