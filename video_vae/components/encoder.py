"""
This file contains the encoder part of the model.
"""
import torch.nn as nn
import torch.nn.functional as F


class ConvEncoder16x16(nn.Module):
    """
    A 16x16 convolutional encoder that extracts features from input images.

    This encoder uses a series of convolutional layers with Group Normalization
    and GELU activation to process an input image and project it into a
    feature space of a specified dimension.
    """
    def __init__(self, n_c=512):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),  # 8 groups of 8 channels each
            nn.GELU(approximate='tanh'),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(16, 128),  # 16 groups × 8 ch
            nn.GELU(approximate='tanh'),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, 256),  # 32 groups × 8 ch
            nn.GELU(approximate='tanh'),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(64, 512),  # 64 groups × 8 ch
            nn.GELU(approximate='tanh'),
        )
        self.resizer = lambda x: F.interpolate(x, size=(32, 32), mode='bilinear', align_corners=False)
        self.project = nn.Conv2d(512, n_c, kernel_size=1, bias=True)

    def forward(self, x):
        """
        Forward pass of the convolutional encoder.

        Args:
            x (torch.Tensor): The input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: The encoded output tensor.
        """
        x = self.features(x)
        x = self.resizer(x)
        x = self.project(x)
        return x