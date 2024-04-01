import importlib
from os import path as osp
import torch

from src.utils import scandir

# automatically scan and import model modules
# scan all the files under the 'models' folder and collect files ending with
# '_model.py'
model_folder = osp.dirname(osp.abspath(__file__))
model_filenames = [
    osp.splitext(osp.basename(v))[0]
    for v in scandir(model_folder)
    if v.endswith("_model.py")
]
# import all the model modules
_model_modules = [
    importlib.import_module(f"src.models.{file_name}") for file_name in model_filenames
]


def create_model(opt):
    """Create model.

    Args:
        opt (dict): Configuration. It contains:
            model_type (str): Model type.
    """
    model_type = opt["model_type"]

    # dynamic instantiation
    for module in _model_modules:
        model_cls = getattr(module, model_type, None)
        if model_cls is not None:
            break
    if model_cls is None:
        raise ValueError(f"Model {model_type} is not found.")

    model = model_cls(opt)

    print(f"Model [{model.__class__.__name__}] is created.")

    # remain object in the code for future applications...
    model = model.net_g

    return model.to("cpu")
