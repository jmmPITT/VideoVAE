"""
This file contains the custom LSTM cell for the model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomLSTMCell(nn.Module):
    """
    A custom LSTM cell that integrates a transformer block for state updates.
    """
    def __init__(self, patch_size=12 * 16, d_model=64, dff=256, noise_std=0.005, attention_block=None):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        self.dff = dff
        self.noise_std = noise_std
        self.attention_block = attention_block

        # LSTM gates on hidden state
        self.WI1 = nn.Linear(dff, dff)
        self.WI2 = nn.Linear(dff, dff)
        self.WF1 = nn.Linear(dff, dff)
        self.WF2 = nn.Linear(dff, dff)
        self.WO1 = nn.Linear(dff, dff)
        self.WO2 = nn.Linear(dff, dff)
        self.WZ1 = nn.Linear(dff, dff)
        self.WZ2 = nn.Linear(dff, dff)

        # LSTM gates on input
        self.RI1 = nn.Linear(d_model, dff)
        self.RI2 = nn.Linear(d_model, dff)
        self.RF1 = nn.Linear(d_model, dff)
        self.RF2 = nn.Linear(d_model, dff)
        self.RO1 = nn.Linear(dff, dff)
        self.RO2 = nn.Linear(dff, dff)
        self.RZ1 = nn.Linear(d_model, dff)
        self.RZ2 = nn.Linear(d_model, dff)

        self.normC1 = nn.LayerNorm(dff)
        self.normC2 = nn.LayerNorm(dff)

    def _lstm_step(self, Zi, Ci, *context_tensors):
        """
        Internal function to compute a single LSTM step.
        """
        Zi = Zi.view(-1, self.patch_size, self.d_model)
        Ci = Ci.view(-1, self.patch_size, self.dff)

        noise_C1 = torch.randn_like(Ci) * self.noise_std
        noise_C2 = torch.randn_like(Ci) * self.noise_std
        noise_Z1 = torch.randn_like(Zi) * self.noise_std
        noise_Z2 = torch.randn_like(Zi) * self.noise_std
        C_noisy1 = Ci + noise_C1
        C_noisy2 = Ci + noise_C2
        Z_noisy1 = Zi + noise_Z1
        Z_noisy2 = Zi + noise_Z2

        I_tilde = self.WI1(C_noisy1) + self.RI1(Z_noisy1) + self.WI2(C_noisy2) + self.RI2(Z_noisy2)
        F_tilde = self.WF1(C_noisy1) + self.RF1(Z_noisy1) + self.WF2(C_noisy2) + self.RF2(Z_noisy2)
        Z_tilde = self.WZ1(C_noisy1) + self.RZ1(Z_noisy1) + self.WZ2(C_noisy2) + self.RZ2(Z_noisy2)

        I_tilde = F.gelu(I_tilde)
        F_tilde = F.gelu(F_tilde)
        I_t = torch.exp(-I_tilde)
        F_t = torch.exp(-F_tilde)

        Z_t = Z_tilde

        C_t = self.normC1(Ci * F_t + Z_t * I_t)

        if self.attention_block:
            C_t_ = self.attention_block(C_t, *context_tensors)[0]
        else:
            C_t_ = C_t

        noise_C1 = C_t + torch.randn_like(C_t) * self.noise_std
        noise_C2 = C_t + torch.randn_like(C_t) * self.noise_std
        noise_C1_ = C_t_ + torch.randn_like(C_t_) * self.noise_std
        noise_C2_ = C_t_ + torch.randn_like(C_t_) * self.noise_std
        O_tilde = self.WO1(noise_C1) + self.RO1(noise_C1_) + self.WO2(noise_C2) + self.RO2(noise_C2_)
        O_t = torch.sigmoid(O_tilde.clamp(min=-20, max=20))

        C_t = self.normC2(O_t * C_t + (1 - O_t) * C_t_)
        return C_t

    def forward(self, Zi, Ci, *context_tensors):
        """
        Forward pass of the custom LSTM cell.

        Args:
            Zi (torch.Tensor): The input tensor.
            Ci (torch.Tensor): The cell state tensor.
            *context_tensors (torch.Tensor): Optional context tensors for the attention block.

        Returns:
            torch.Tensor: The updated cell state.
        """
        return self._lstm_step(Zi, Ci, *context_tensors)