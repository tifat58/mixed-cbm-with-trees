import torch
import torch.nn as nn
import numpy as np

from torchvision import transforms


class ResNetBottom(nn.Module):
    def __init__(self, original_model):
        super(ResNetBottom, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-1])
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x

def resNetBottomWithLL(original_model):

    if hasattr(original_model, 'output'):
        original_model.output = torch.nn.Linear(512, 112)
    return original_model

class ResNetTop(nn.Module):
    def __init__(self, original_model):
        super(ResNetTop, self).__init__()
        self.features = nn.Sequential(*[list(original_model.children())[-1]])

    def forward(self, x):
        x = self.features(x)
        x = nn.Softmax(dim=-1)(x)
        return x


def get_model(out_dir, backbone_name="resnet18_cub", full_model=True):

    if backbone_name == "resnet18_cub":
        from pytorchcv.model_provider import get_model as ptcv_get_model
        model = ptcv_get_model(backbone_name, pretrained=True, root=out_dir)
        model, backbone, model_top = resNetBottomWithLL(model), ResNetBottom(model), ResNetTop(model)
        cub_mean_pxs = np.array([0.5, 0.5, 0.5])
        cub_std_pxs = np.array([2., 2., 2.])
        preprocess = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(cub_mean_pxs, cub_std_pxs)
            ])
    else:
        raise ValueError(backbone_name)

    if full_model:
        return model, backbone, preprocess
    else:
        return backbone, preprocess


