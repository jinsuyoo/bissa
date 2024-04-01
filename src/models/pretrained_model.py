import importlib
import torch
from copy import deepcopy

from src.models.archs import define_network
from src.models.base_model import BaseModel


class PretrainedModel(BaseModel):
    """Pretrained SR model for self-supervised adaptation."""

    def __init__(self, opt):
        super(PretrainedModel, self).__init__(opt)

        # define network
        self.net_g = define_network(deepcopy(opt["network_g"]))
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt["path"].get("pretrain_network_g", None)
        if load_path is not None:
            if opt["network_g"]["type"] == "RBPN":
                self.net_g = torch.nn.DataParallel(self.net_g, device_ids=[0])
                # self.net_g.cuda()
                self.net_g.load_state_dict(torch.load(load_path))
            else:
                self.load_network(
                    self.net_g, load_path, self.opt["path"].get("strict_load_g", True)
                )
