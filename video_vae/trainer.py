"""
This file contains the trainer for the VideoVAE model.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
import torch.cuda as cuda
import gc
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

from .model import TransformerNetwork


class PerceptualLoss(nn.Module):
    """
    Calculates perceptual loss using a pre-trained VGG network.

    This loss function compares the high-level features of two images
    extracted from a VGG16 model, which can lead to more visually
    pleasing results than traditional pixel-wise losses.
    """
    def __init__(self, resize=True, normalize=True):
        super(PerceptualLoss, self).__init__()
        weights = models.VGG16_Weights.IMAGENET1K_V1
        self.vgg = models.vgg16(weights=weights).features[:29].eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

        self.resize = resize
        self.normalize = normalize
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)) if resize else nn.Identity(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) if normalize else nn.Identity()
        ])
        self.criterion = nn.MSELoss()

    def forward(self, x, y):
        """
        Calculates the perceptual loss between two tensors.

        Args:
            x (torch.Tensor): The first input tensor.
            y (torch.Tensor): The second input tensor.

        Returns:
            torch.Tensor: The perceptual loss.
        """
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
            y = y.repeat(1, 3, 1, 1)

        if self.resize or self.normalize:
            x = self.transform(x)
            y = self.transform(y)

        x_features = self.vgg(x)
        y_features = self.vgg(y)

        return self.criterion(x_features, y_features)


class MovingMNISTDataset(Dataset):
    """
    A custom dataset for loading and processing the Moving MNIST dataset.

    This dataset loads video sequences from a .npy file, normalizes them,
    and splits them into patches for use in the model.
    """
    def __init__(self, npy_file):
        self.data = np.load(npy_file, mmap_mode='r')
        self.num_sequences, self.num_frames, H, W, self.channels = self.data.shape
        self.patch_h, self.patch_w = 15, 20
        self.grid_h, self.grid_w = H // self.patch_h, W // self.patch_w
        self.num_patches = self.grid_h * self.grid_w

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        video = (self.data[idx].astype(np.float32) / 255.0)
        v = video.reshape(
            self.num_frames,
            self.grid_h, self.patch_h,
            self.grid_w, self.patch_w,
            self.channels
        )
        v = v.transpose(0, 1, 3, 2, 4, 5)
        patches = v.reshape(
            self.num_frames,
            self.num_patches,
            self.patch_h,
            self.patch_w,
            self.channels
        )
        return torch.from_numpy(patches)


class TemporalAutoencoderTrainer:
    """
    A trainer for the temporal autoencoder model.

    This class handles the training loop, including data loading,
    model optimization, and logging.
    """
    def __init__(self, transformer_model, dataset, batch_size=16, learning_rate=1e-3,
                 num_epochs=10):
        self.transformer_model = transformer_model
        self.dataset = dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transformer_model.to(self.device)
        self.optimizer_TAE = optim.Adam(self.transformer_model.parameters(), lr=self.learning_rate, betas=(0.9, 0.99),
                                eps=1e-8, weight_decay=1e-4)
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def train(self):
        for epoch in range(1, self.num_epochs + 1):
            epoch_loss = 0.0
            for batch_idx, sequences in enumerate(self.dataloader):
                sequences = sequences.to(self.device)
                batch_size, num_frames, num_patch, height, width, channels = sequences.size()
                latent_sequences = []

                with torch.no_grad():
                    for i in range(num_frames):
                        frames = sequences[:, i, :, :, :].view(-1, 256, 15 * 20 * 3)
                        latent_sequences.append(frames.unsqueeze(1))

                latent_sequences = torch.cat(latent_sequences, dim=1)
                total_loss = 0.0
                C_F1 = torch.zeros(batch_size, self.transformer_model.num_patch_one, self.transformer_model.dff1, requires_grad=True).to(self.device)
                C_F2 = torch.zeros(batch_size, self.transformer_model.num_patch_two, self.transformer_model.dff2, requires_grad=True).to(self.device)
                C_F3 = torch.zeros(batch_size, self.transformer_model.num_patch_three, self.transformer_model.dff3, requires_grad=True).to(self.device)
                C_F4 = torch.zeros(batch_size, self.transformer_model.num_patch_four, self.transformer_model.dff4, requires_grad=True).to(self.device)

                state = latent_sequences.view(batch_size, 50, 256, 15 * 20 * 3)
                total_loss = 0
                recon_loss = 0
                C1_Forward_List = []
                C2_Forward_List = []
                C3_Forward_List = []
                C4_Forward_List = []

                for t in range(50):
                    input_state = state[:, t, :].view(-1, 256, 15 * 20 * 3)
                    Z, C_F1, C_F2, C_F3, C_F4 = self.transformer_model.encoder(
                        input_state, C_F1, C_F2, C_F3, C_F4, t)

                    C1_Forward_List.append(C_F1.view(-1, 1, self.transformer_model.num_patch_one, self.transformer_model.dff1))
                    C2_Forward_List.append(C_F2.view(-1, 1, self.transformer_model.num_patch_two, self.transformer_model.dff2))
                    C3_Forward_List.append(C_F3.view(-1, 1, self.transformer_model.num_patch_three, self.transformer_model.dff3))
                    C4_Forward_List.append(C_F4.view(-1, 1, self.transformer_model.num_patch_four, self.transformer_model.dff4))

                    if t == 49:
                        out, C1_backwards, C2_backwards, C3_backwards, C4_backwards, mu1, log_var1, mu2, log_var2, mu3, log_var3, mu4, log_var4 = self.transformer_model.decoder(Z, C_F1, C_F2, C_F3, C_F4, t + 5)

                        kl_loss = -0.5 * torch.mean(1 + log_var1 - mu1.pow(2) - log_var1.exp())
                        kl_loss += -0.5 * torch.mean(1 + log_var2 - mu2.pow(2) - log_var2.exp())
                        kl_loss += -0.5 * torch.mean(1 + log_var3 - mu3.pow(2) - log_var3.exp())
                        kl_loss += -0.5 * torch.mean(1 + log_var4 - mu4.pow(2) - log_var4.exp())

                        z_target_ = state.view(batch_size, num_frames, 256, 15 * 20 * 3)
                        z_target_ = z_target_.view(batch_size, num_frames, 16, 16, 15, 20, 3)
                        z_target_ = z_target_.permute(0, 1, 2, 4, 3, 5, 6).contiguous()
                        z_target_ = z_target_.view(batch_size, 50, 240, 320, 3)
                        out = out.view(-1, 55, 240, 320, 3)

                        recon_loss += F.mse_loss(out[:, 0:50], z_target_[:, 0:50])
                        total_loss += recon_loss * 10.0 + kl_loss * 0.01

                        self.optimizer_TAE.zero_grad()
                        total_loss.backward()
                        self.optimizer_TAE.step()

                if batch_idx % 10 == 0:
                    print(
                        f'Epoch [{epoch}/{self.num_epochs}] Batch [{batch_idx}/{len(self.dataloader)}] '
                        f'TAELoss: {total_loss.item():.6f}', f'recon_loss: {recon_loss.item():.6f}', f'kl_loss: {kl_loss.item():.6f}')
                if batch_idx % 100 == 0:
                    self.save_model_TAE(epoch)

    def save_model_TAE(self, epoch):
        model_save_path = f'transformer_model.pth'
        torch.save(self.transformer_model.state_dict(), model_save_path)
        print(f'Model saved at {model_save_path}')

    def load_model(self, epoch):
        model_save_path = f'transformer_model_epoch_{epoch}.pth'
        self.transformer_model.load_state_dict(torch.load(model_save_path, map_location=self.device))
        self.transformer_model.to(self.device)
        self.transformer_model.eval()
        print(f'Model loaded from {model_save_path}')