from typing import Optional

import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F

from torchvision.models.resnet import resnet18, resnet50, wide_resnet50_2


class SimCLR(nn.Module):
    def __init__(self,
                 hidden_dim: int = 128,
                 model_name: str = 'resnet18',
                 encoder_dim: Optional[int] = None,
                 use_deeper_proj_head: Optional[bool] = False,
                 normalize: bool = False
                 ):
        super().__init__()
        self.encoder_dim = encoder_dim if encoder_dim is not None else 4 * hidden_dim
        self.normalize = normalize
        # Base model f(.)
        if model_name == 'resnet18':
            self.convnet = resnet18(
                pretrained=False, num_classes=self.encoder_dim
            )  # num_classes is the output size of the last linear layer
        elif model_name == 'resnet50':
            self.convnet = resnet50(
                pretrained=False, num_classes=self.encoder_dim
            )
        elif model_name == 'wide_resnet50_2':
            self.convnet = wide_resnet50_2(pretrained=False, num_classes=self.encoder_dim)
        # setting projection head
        if use_deeper_proj_head is not True:
            # The MLP for g(.) consists of Linear->ReLU->Linear
            self.convnet.fc = nn.Sequential(
                self.convnet.fc,  # Linear(ResNet output, encoder_dim)
                nn.ReLU(inplace=True),
                nn.Linear(self.encoder_dim, hidden_dim),
            )
        else:
            # inspired from https://github.com/ae-foster/pytorch-simclr/blob/simclr-master/critic.py
            # The MLP for g(.) consists of
            # Linear->BatchNorm1d->ReLU->
            # Linear->BatchNorm1d->ReLU->
            # Linear->BatchNorm1d
            self.convnet.fc = nn.Sequential(
                self.convnet.fc,  # Linear(ResNet output, encoder_dim)
                nn.BatchNorm1d(self.encoder_dim),
                nn.ReLU(inplace=True),
                nn.Linear(encoder_dim, encoder_dim, bias=False),
                nn.BatchNorm1d(self.encoder_dim),
                nn.ReLU(inplace=True),
                nn.Linear(encoder_dim, hidden_dim, bias=False),
                nn.BatchNorm1d(hidden_dim, affine=False),
            )

    def forward(self, x):
        output = self.convnet(x)
        if self.normalize:
            output = F.normalize(output)
        return output


if __name__ == '__main__':
    num_input_channels, width, height = 3, 224, 224
    model = SimCLR(hidden_dim=128).to('cuda')
    summary(model, input_size=(num_input_channels, height, width))
