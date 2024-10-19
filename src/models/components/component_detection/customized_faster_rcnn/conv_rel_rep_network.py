from typing import Optional

from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn
import torch.nn.functional as F
import torch


class ConvRelationRepresentationNetwork(nn.Module):
    def __init__(self,
                 input_channels=256,
                 feature_dim=64,
                 representation_size=128):
        super(ConvRelationRepresentationNetwork, self).__init__()

        def create_feature_processor():
            return nn.Sequential(
                nn.Conv2d(input_channels, feature_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(feature_dim),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
                nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(feature_dim),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                # nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
                # nn.BatchNorm2d(feature_dim),
                # nn.ReLU(inplace=True),
                # nn.MaxPool2d(kernel_size=2, stride=2) # [B, 64, 1, 1]
            )

        # convolutional layers to process speech bubble features
        self.conv_speech = create_feature_processor()
        # convolutional layers to process face features
        self.conv_face = create_feature_processor()
        # convolutional layers to process encapsulating box features
        self.conv_box = create_feature_processor()

        # convolutional layers to process combined features
        self.conv_combined = nn.Sequential(
            nn.Conv2d(feature_dim * 3, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            # nn.BatchNorm2d(feature_dim),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            # nn.BatchNorm2d(feature_dim),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc = nn.Sequential(
            nn.Linear(feature_dim, representation_size),
            nn.ReLU(inplace=True),
            # nn.Linear(1024, representation_size),
            # nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor):
        box, speech, face = x[:, 0], x[:, 1], x[:, 2]

        # process speech bubble features
        speech_out = self.conv_speech(speech)
        # speech_out = speech_out.view(speech_out.size(0), -1)

        # process face features
        face_out = self.conv_face(face)
        # face_out = face_out.view(face_out.size(0), -1)

        # process encapsulating box features
        box_out = self.conv_box(box)
        # box_out = box_out.view(box_out.size(0), -1)

        # concatenate the features
        combined_out = torch.cat((speech_out, face_out, box_out), dim=1)
        # combined_out = combined_out.view(-1, self.feature_dim * 3, 4, 4)

        # process the combined features
        combined_out = self.conv_combined(combined_out)
        combined_out = combined_out.view(combined_out.size(0), -1)
        output = self.fc(combined_out)

        return output
