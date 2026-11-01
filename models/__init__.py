"""Model definitions (YOLO26 / YOLO26-Depth builders, CA, CBAM, KD wrappers).

Importing this package registers the custom attention modules with the Ultralytics YAML parser, which is
also required to unpickle checkpoints of attention-augmented teachers.
"""

from src.utils.versions import check_ultralytics_version

check_ultralytics_version()  # fail fast with a clear message on an unsupported Ultralytics release

from models.common.model_utils import register_custom_modules  # noqa: E402

register_custom_modules()
