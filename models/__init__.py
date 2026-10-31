"""Model definitions (YOLO26 / YOLO26-Depth builders, CA, CBAM, KD wrappers).

Importing this package registers the custom attention modules with the Ultralytics YAML parser, which is
also required to unpickle checkpoints of attention-augmented teachers.
"""

from models.common.model_utils import register_custom_modules

register_custom_modules()
