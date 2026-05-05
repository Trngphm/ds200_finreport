
import importlib
import os
from builders.registry import MODEL_REGISTRY


def _auto_import_models():
    """
    FIX: import từ thư mục `modules/` (không phải `models/`)
         và match pattern `module*.py` thay vì `*_model.py`.
    """
    modules_dir = os.path.join(os.path.dirname(__file__), '..', 'modules')
    for fname in os.listdir(modules_dir):
        if fname.endswith('.py') and not fname.startswith('_'):
            module_name = f"modules.{fname[:-3]}"
            importlib.import_module(module_name)


def build_model(cfg):
    _auto_import_models()
    name = cfg.model.name
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Model '{name}' chưa được đăng ký. "
            f"Các model hiện có: {list(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[name](cfg.model)