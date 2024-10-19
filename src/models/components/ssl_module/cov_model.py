import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torchvision.models.resnet import resnet18, resnet50
import torch


class CovModel6(nn.Module):
    def __init__(self,
                 model_name: str = 'resnet18',
                 projector: str = '4096-4096-128',
                 normalize_on: bool = False):
        super().__init__()
        if model_name == "resnet18":
            self.backbone = resnet18(zero_init_residual=True)
            self.backbone.fc = nn.Identity()
        elif model_name == "resnet50":
            self.backbone = resnet50(zero_init_residual=True)
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError("given setting has not included yet!")

        # projector
        if model_name == "resnet18":
            sizes = [512] + list(map(int, projector.split('-')))  # 512 = self.feature_dim
        elif model_name == "resnet50":
            sizes = [2048] + list(map(int, projector.split('-')))  # 512 = self.feature_dim
        else:
            raise ValueError("given setting has not included yet!")
        layers = []

        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)
        self.normalized = normalize_on
        print(self.normalized)

    def forward(self, y, is_ssl: bool = True):
        feature_all = self.backbone(y)
        z_all = self.projector(feature_all)

        if self.normalized:
            if is_ssl:
                z1, z2 = torch.tensor_split(z_all, 2)
                z1 = F.normalize(z1, p=2)
                z2 = F.normalize(z2, p=2)
                z_all = torch.cat([z1, z2])
            else:
                z_all = F.normalize(z_all, p=2)

        return z_all


if __name__ == '__main__':
    num_input_channels, width, height = 3, 224, 224
    model = CovModel6(normalize_on=True, model_name='resnet50').to('cuda')
    summary(model, input_size=(num_input_channels, height, width))
