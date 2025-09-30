
import os
import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch.nn.utils.rnn as rnn_utils
import torch.utils.checkpoint as checkpoint
import math
from RVITS import *
# torch.backends.cudnn.enabled = False



class ConvEncoder16x16(nn.Module):
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
        self.resizer = lambda x: F.interpolate(x, size=(32, 32), mode='bilinear', align_corners=False) # <--- ADD A RESIZER
        # self.pool = nn.AdaptiveAvgPool2d((32, 32))
        self.project = nn.Conv2d(512, n_c, kernel_size=1,bias=True)

    def forward(self, x):
        x = self.features(x)
        x = self.resizer(x)
        x = self.project(x)
        return x


class TransformerBlockM0(nn.Module):
    """
    Transformer block that processes input state with self-attention mechanism.
    Incorporates hidden states from LSTMs for additional context.
    """

    def __init__(self, input_dims=64, patch_length=12 * 16 + 1, dff=256, dropout=0.1, num_heads=2):
        super(TransformerBlockM0, self).__init__()
        self.input_dims = input_dims  # Using reduced dimension from CNN (256)
        self.patch_length = patch_length  # Number of patches (reduced from 256)
        self.d_model = self.input_dims  # Model dimension
        self.num_heads = num_heads  # Increased number of attention heads (works well with 256 dim)
        self.dff = dff  # Feed-forward network dimension
        self.dropout = dropout  # Dropout rate

        # Calculate key dimension (must be divisible by num_heads)
        self.d_k = self.d_model // self.num_heads

        # Query, key, and value projections for attention
        self.W_q = nn.Linear(self.d_model, self.d_k * self.num_heads)
        self.W_k = nn.Linear(self.d_model, self.d_k * self.num_heads)
        self.W_v = nn.Linear(self.d_model, self.d_k * self.num_heads)

        # Layer normalization for attention components
        self.normQE = nn.LayerNorm(self.d_model)
        self.normKE = nn.LayerNorm(self.d_model)
        self.normVE = nn.LayerNorm(self.d_model)

        # Feed-forward network layers
        self.linear1 = nn.Linear(self.num_heads * self.d_k, self.d_model * 4)
        self.linear2 = nn.Linear(self.d_model * 4, self.d_model)
        self.dropout1 = nn.Dropout(self.dropout)

        # Normalization and dropout for feed-forward path
        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)
        self.dropout2 = nn.Dropout(self.dropout)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        """
        Forward pass through the transformer block.

        Args:
            state: Input state tensor
            H: Hidden state from LSTM

        Returns:
            Z5: Processed state tensor
        """
        # Reshape input state
        # print('ayo',state.shape)

        state = state.view(-1, self.patch_length, self.input_dims)
        # print('ayo',state.shape)
        # Reshape hidden state for context
        batch_size = state.shape[0]

        # Attention mechanism
        # Construct query, key, and value with both input state and hidden state context
        q = self.normQE(self.W_q(state)).view(batch_size, -1, self.num_heads, self.d_k)
        k = self.normKE(self.W_k(state)).view(batch_size, -1, self.num_heads, self.d_k)
        v = self.normVE(self.W_v(state)).view(batch_size, -1, self.num_heads, self.d_k)

        # Reshape for multi-head attention
        q, k, v = [x.transpose(1, 2) for x in (q, k, v)]  # [batch_size, num_heads, seq_len, d_k]

        # Compute attention and get values
        attn_values, attn_weights = self.calculate_attention(q, k, v)
        attn_values = attn_values.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)

        # Residual connection and dropout
        Z1 = state + self.dropout1(attn_values)

        # Feed-forward network with residual connection
        Z2_norm = self.norm1(Z1)
        Z3 = F.gelu(self.linear1(Z2_norm),approximate='tanh')
        Z4 = self.linear2(Z3)
        dZ = self.dropout1(attn_values) + self.dropout2(Z4)
        Z5 = state + dZ

        return Z5, dZ, state

    # RVITS.py, in TransformerBlockX.calculate_attention
    def calculate_attention(self, q, k, v):
        """
        Calculate scaled dot-product attention using fused kernels (SDPA) for speedup.
        """
        # NO NEED FOR MANUAL SCALING OR SOFTMAX

        # Use fused SDPA. SDPA automatically handles the scaling by sqrt(d_k)
        # We assume no attention mask is needed here (is_causal=False, attn_mask=None)
        attention_output = F.scaled_dot_product_attention(
            query=q, 
            key=k, 
            value=v, 
            attn_mask=None, 
            dropout_p=0.0, # Dropout should be applied *after* this fusion 
                        # but can be added here if needed; setting to 0.0 for now
            is_causal=False
        )

        # NOTE: SDPA does not return the weights (attention_weights). 
        # If you require the weights for interpretability/logging, you must keep 
        # the original code. For pure speedup, this is the best path.

        # For compatibility with your current function signature:
        # Since SDPA doesn't return weights, we return None for attention_weights 
        # as it is unused in the TransformerBlockX.forward methods.
        attention_weights = None 

        return attention_output, attention_weights


class TransformerBlockM1(nn.Module):
    """
    Transformer block that processes input state with self-attention mechanism.
    Incorporates hidden states from LSTMs for additional context.
    """

    def __init__(self, input_dims=64, patch_length=12 * 16 + 1, dff=256, dropout=0.1, num_heads=2):
        super(TransformerBlockM1, self).__init__()
        self.input_dims = input_dims  # Using reduced dimension from CNN (256)
        self.patch_length = patch_length  # Number of patches (reduced from 256)
        self.d_model = self.input_dims  # Model dimension
        self.num_heads = num_heads  # Increased number of attention heads (works well with 256 dim)
        self.dff = dff  # Feed-forward network dimension
        self.dropout = dropout  # Dropout rate

        # Calculate key dimension (must be divisible by num_heads)
        self.d_k = self.d_model // self.num_heads

        # Query, key, and value projections for attention
        self.W_q = nn.Linear(self.d_model, self.d_k * self.num_heads)
        self.W_k = nn.Linear(self.d_model, self.d_k * self.num_heads)
        self.W_v = nn.Linear(self.d_model, self.d_k * self.num_heads)

        # Context projections for hidden states
        self.W_C1q = nn.Linear(self.dff, self.d_k * self.num_heads)
        self.W_C1k = nn.Linear(self.dff, self.d_k * self.num_heads)
        self.W_C1v = nn.Linear(self.dff, self.d_k * self.num_heads)

        # Layer normalization for attention components
        self.normQE = nn.LayerNorm(self.d_model)
        self.normKE = nn.LayerNorm(self.d_model)
        self.normVE = nn.LayerNorm(self.d_model)
        self.normQH = nn.LayerNorm(self.d_model)
        self.normKH = nn.LayerNorm(self.d_model)
        self.normVH = nn.LayerNorm(self.d_model)

        # Feed-forward network layers
        self.linear1 = nn.Linear(self.num_heads * self.d_k, self.d_model * 4)
        self.linear2 = nn.Linear(self.d_model * 4, self.d_model)
        self.dropout1 = nn.Dropout(self.dropout)

        # Normalization and dropout for feed-forward path
        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)
        self.dropout2 = nn.Dropout(self.dropout)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, C):
        """
        Forward pass through the transformer block.

        Args:
            state: Input state tensor
            H: Hidden state from LSTM

        Returns:
            Z5: Processed state tensor
        """
        # Reshape input state
        # print('ayo',state.shape)

        state = state.view(-1, self.patch_length, self.input_dims)
        # print('ayo',state.shape)
        # Reshape hidden state for context
        batch_size = state.shape[0]

        # Attention mechanism
        # Construct query, key, and value with both input state and hidden state context
        q = self.normQE(self.W_q(state)).view(batch_size, -1, self.num_heads, self.d_k) * self.normQH(
            self.W_C1q(C)).view(batch_size, -1, self.num_heads, self.d_k)
        k = self.normKE(self.W_k(state)).view(batch_size, -1, self.num_heads, self.d_k) * self.normKH(
            self.W_C1k(C)).view(batch_size, -1, self.num_heads, self.d_k)
        v = self.normVE(self.W_v(state)).view(batch_size, -1, self.num_heads, self.d_k) * self.normVH(
            self.W_C1v(C)).view(batch_size, -1, self.num_heads, self.d_k)

        # Reshape for multi-head attention
        q, k, v = [x.transpose(1, 2) for x in (q, k, v)]  # [batch_size, num_heads, seq_len, d_k]

        # Compute attention and get values
        attn_values, attn_weights = self.calculate_attention(q, k, v)
        attn_values = attn_values.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)

        # Residual connection and dropout
        Z1 = state + self.dropout1(attn_values)

        # Feed-forward network with residual connection
        Z2_norm = self.norm1(Z1)
        Z3 = F.gelu(self.linear1(Z2_norm),approximate='tanh')
        Z4 = self.linear2(Z3)
        dZ = self.dropout1(attn_values) + self.dropout2(Z4)
        Z5 = state + dZ

        return Z5, dZ, state

    # RVITS.py, in TransformerBlockX.calculate_attention
    def calculate_attention(self, q, k, v):
        """
        Calculate scaled dot-product attention using fused kernels (SDPA) for speedup.
        """
        # NO NEED FOR MANUAL SCALING OR SOFTMAX

        # Use fused SDPA. SDPA automatically handles the scaling by sqrt(d_k)
        # We assume no attention mask is needed here (is_causal=False, attn_mask=None)
        attention_output = F.scaled_dot_product_attention(
            query=q, 
            key=k, 
            value=v, 
            attn_mask=None, 
            dropout_p=0.0, # Dropout should be applied *after* this fusion 
                        # but can be added here if needed; setting to 0.0 for now
            is_causal=False
        )

        # NOTE: SDPA does not return the weights (attention_weights). 
        # If you require the weights for interpretability/logging, you must keep 
        # the original code. For pure speedup, this is the best path.

        # For compatibility with your current function signature:
        # Since SDPA doesn't return weights, we return None for attention_weights 
        # as it is unused in the TransformerBlockX.forward methods.
        attention_weights = None 

        return attention_output, attention_weights


class TransformerBlockM2(nn.Module):
    """
    Transformer block that processes input state with self-attention mechanism.
    Incorporates hidden states from LSTMs for additional context.
    """

    def __init__(self, input_dims=64, patch_length=12 * 16 + 1, dff=256, dropout=0.1, num_heads=2):
        super(TransformerBlockM2, self).__init__()
        self.input_dims = input_dims  # Using reduced dimension from CNN (256)
        self.patch_length = patch_length  # Number of patches (reduced from 256)
        self.d_model = self.input_dims  # Model dimension
        self.num_heads = num_heads  # Increased number of attention heads (works well with 256 dim)
        self.dff = dff  # Feed-forward network dimension
        self.dropout = dropout  # Dropout rate

        # Calculate key dimension (must be divisible by num_heads)
        self.d_k = self.d_model // self.num_heads

        # Query, key, and value projections for attention
        self.W_q = nn.Linear(self.d_model, self.d_k * self.num_heads)
        self.W_k = nn.Linear(self.d_model, self.d_k * self.num_heads)
        self.W_v = nn.Linear(self.d_model, self.d_k * self.num_heads)

        # Context projections for hidden states
        self.W_C1q = nn.Linear(self.dff * 2, self.d_k * self.num_heads)
        self.W_C1k = nn.Linear(self.dff * 2, self.d_k * self.num_heads)
        self.W_C1v = nn.Linear(self.dff * 2, self.d_k * self.num_heads)

        # Layer normalization for attention components
        self.normQE = nn.LayerNorm(self.d_model)
        self.normKE = nn.LayerNorm(self.d_model)
        self.normVE = nn.LayerNorm(self.d_model)
        self.normQH = nn.LayerNorm(self.d_model)
        self.normKH = nn.LayerNorm(self.d_model)
        self.normVH = nn.LayerNorm(self.d_model)

        # Feed-forward network layers
        self.linear1 = nn.Linear(self.num_heads * self.d_k, self.d_model * 4)
        self.linear2 = nn.Linear(self.d_model * 4, self.d_model)
        self.dropout1 = nn.Dropout(self.dropout)

        # Normalization and dropout for feed-forward path
        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)
        self.dropout2 = nn.Dropout(self.dropout)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, C1, C2):
        """
        Forward pass through the transformer block.

        Args:
            state: Input state tensor
            H: Hidden state from LSTM

        Returns:
            Z5: Processed state tensor
        """
        # Reshape input state
        # print('ayo',state.shape)

        state = state.view(-1, self.patch_length, self.input_dims)
        # print('ayo',state.shape)
        # Reshape hidden state for context
        batch_size = state.shape[0]

        C1 = C1.view(-1, self.patch_length, self.dff)
        C2 = C2.view(-1, self.patch_length, self.dff)
        C = torch.cat((C1, C2), dim=-1)
        # Attention mechanism
        # Construct query, key, and value with both input state and hidden state context
        q = self.normQE(self.W_q(state)).view(batch_size, -1, self.num_heads, self.d_k) * self.normQH(
            self.W_C1q(C)).view(batch_size, -1, self.num_heads, self.d_k)
        k = self.normKE(self.W_k(state)).view(batch_size, -1, self.num_heads, self.d_k) * self.normKH(
            self.W_C1k(C)).view(batch_size, -1, self.num_heads, self.d_k)
        v = self.normVE(self.W_v(state)).view(batch_size, -1, self.num_heads, self.d_k) * self.normVH(
            self.W_C1v(C)).view(batch_size, -1, self.num_heads, self.d_k)

        # Reshape for multi-head attention
        q, k, v = [x.transpose(1, 2) for x in (q, k, v)]  # [batch_size, num_heads, seq_len, d_k]

        # Compute attention and get values
        attn_values, attn_weights = self.calculate_attention(q, k, v)
        attn_values = attn_values.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)

        # Residual connection and dropout
        Z1 = state + self.dropout1(attn_values)

        # Feed-forward network with residual connection
        Z2_norm = self.norm1(Z1)
        Z3 = F.gelu(self.linear1(Z2_norm),approximate='tanh')
        Z4 = self.linear2(Z3)
        dZ = self.dropout1(attn_values) + self.dropout2(Z4)
        Z5 = state + dZ

        return Z5, dZ, state

    # RVITS.py, in TransformerBlockX.calculate_attention
    def calculate_attention(self, q, k, v):
        """
        Calculate scaled dot-product attention using fused kernels (SDPA) for speedup.
        """
        # NO NEED FOR MANUAL SCALING OR SOFTMAX

        # Use fused SDPA. SDPA automatically handles the scaling by sqrt(d_k)
        # We assume no attention mask is needed here (is_causal=False, attn_mask=None)
        attention_output = F.scaled_dot_product_attention(
            query=q, 
            key=k, 
            value=v, 
            attn_mask=None, 
            dropout_p=0.0, # Dropout should be applied *after* this fusion 
                        # but can be added here if needed; setting to 0.0 for now
            is_causal=False
        )

        # NOTE: SDPA does not return the weights (attention_weights). 
        # If you require the weights for interpretability/logging, you must keep 
        # the original code. For pure speedup, this is the best path.

        # For compatibility with your current function signature:
        # Since SDPA doesn't return weights, we return None for attention_weights 
        # as it is unused in the TransformerBlockX.forward methods.
        attention_weights = None 

        return attention_output, attention_weights


class TransformerBlockM3(nn.Module):
    """
    Transformer block that processes input state with self-attention mechanism.
    Incorporates hidden states from LSTMs for additional context.
    """

    def __init__(self, input_dims=64, patch_length=12 * 16 + 1, dff=256, dropout=0.1, num_heads=2):
        super(TransformerBlockM3, self).__init__()
        self.input_dims = input_dims  # Using reduced dimension from CNN (256)
        self.patch_length = patch_length  # Number of patches (reduced from 256)
        self.d_model = self.input_dims  # Model dimension
        self.num_heads = num_heads  # Increased number of attention heads (works well with 256 dim)
        self.dff = dff  # Feed-forward network dimension
        self.dropout = dropout  # Dropout rate

        # Calculate key dimension (must be divisible by num_heads)
        self.d_k = self.d_model // self.num_heads

        # Query, key, and value projections for attention
        self.W_q = nn.Linear(self.d_model, self.d_k * self.num_heads)
        self.W_k = nn.Linear(self.d_model, self.d_k * self.num_heads)
        self.W_v = nn.Linear(self.d_model, self.d_k * self.num_heads)

        # Context projections for hidden states
        self.W_C1q = nn.Linear(self.dff * 3, self.d_k * self.num_heads)
        self.W_C1k = nn.Linear(self.dff * 3, self.d_k * self.num_heads)
        self.W_C1v = nn.Linear(self.dff * 3, self.d_k * self.num_heads)

        # Layer normalization for attention components
        self.normQE = nn.LayerNorm(self.d_model)
        self.normKE = nn.LayerNorm(self.d_model)
        self.normVE = nn.LayerNorm(self.d_model)
        self.normQH = nn.LayerNorm(self.d_model)
        self.normKH = nn.LayerNorm(self.d_model)
        self.normVH = nn.LayerNorm(self.d_model)

        # Feed-forward network layers
        self.linear1 = nn.Linear(self.num_heads * self.d_k, self.d_model * 4)
        self.linear2 = nn.Linear(self.d_model * 4, self.d_model)
        self.dropout1 = nn.Dropout(self.dropout)

        # Normalization and dropout for feed-forward path
        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)
        self.dropout2 = nn.Dropout(self.dropout)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, C1, C2, C3):
        """
        Forward pass through the transformer block.

        Args:
            state: Input state tensor
            H: Hidden state from LSTM

        Returns:
            Z5: Processed state tensor
        """
        # Reshape input state
        # print('ayo',state.shape)

        state = state.view(-1, self.patch_length, self.input_dims)
        # print('ayo',state.shape)
        # Reshape hidden state for context
        batch_size = state.shape[0]

        C1 = C1.view(-1, self.patch_length, self.dff)
        C2 = C2.view(-1, self.patch_length, self.dff)
        C3 = C3.view(-1, self.patch_length, self.dff)
        C = torch.cat((C1, C2, C3), dim=-1)
        # Attention mechanism
        # Construct query, key, and value with both input state and hidden state context
        q = self.normQE(self.W_q(state)).view(batch_size, -1, self.num_heads, self.d_k) * self.normQH(
            self.W_C1q(C)).view(batch_size, -1, self.num_heads, self.d_k)
        k = self.normKE(self.W_k(state)).view(batch_size, -1, self.num_heads, self.d_k) * self.normKH(
            self.W_C1k(C)).view(batch_size, -1, self.num_heads, self.d_k)
        v = self.normVE(self.W_v(state)).view(batch_size, -1, self.num_heads, self.d_k) * self.normVH(
            self.W_C1v(C)).view(batch_size, -1, self.num_heads, self.d_k)

        # Reshape for multi-head attention
        q, k, v = [x.transpose(1, 2) for x in (q, k, v)]  # [batch_size, num_heads, seq_len, d_k]

        # Compute attention and get values
        attn_values, attn_weights = self.calculate_attention(q, k, v)
        attn_values = attn_values.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)

        # Residual connection and dropout
        Z1 = state + self.dropout1(attn_values)

        # Feed-forward network with residual connection
        Z2_norm = self.norm1(Z1)
        Z3 = F.gelu(self.linear1(Z2_norm),approximate='tanh')
        Z4 = self.linear2(Z3)
        dZ = self.dropout1(attn_values) + self.dropout2(Z4)
        Z5 = state + dZ

        return Z5, dZ, state

    # RVITS.py, in TransformerBlockX.calculate_attention
    def calculate_attention(self, q, k, v):
        """
        Calculate scaled dot-product attention using fused kernels (SDPA) for speedup.
        """
        # NO NEED FOR MANUAL SCALING OR SOFTMAX

        # Use fused SDPA. SDPA automatically handles the scaling by sqrt(d_k)
        # We assume no attention mask is needed here (is_causal=False, attn_mask=None)
        attention_output = F.scaled_dot_product_attention(
            query=q, 
            key=k, 
            value=v, 
            attn_mask=None, 
            dropout_p=0.0, # Dropout should be applied *after* this fusion 
                        # but can be added here if needed; setting to 0.0 for now
            is_causal=False
        )

        # NOTE: SDPA does not return the weights (attention_weights). 
        # If you require the weights for interpretability/logging, you must keep 
        # the original code. For pure speedup, this is the best path.

        # For compatibility with your current function signature:
        # Since SDPA doesn't return weights, we return None for attention_weights 
        # as it is unused in the TransformerBlockX.forward methods.
        attention_weights = None 

        return attention_output, attention_weights


class TransformerBlockM4(nn.Module):
    """
    Transformer block that processes input state with self-attention mechanism.
    Incorporates hidden states from LSTMs for additional context.
    """

    def __init__(self, input_dims=64, patch_length=12 * 16 + 1, dff=256, dropout=0.1, num_heads=2):
        super(TransformerBlockM4, self).__init__()
        self.input_dims = input_dims  # Using reduced dimension from CNN (256)
        self.patch_length = patch_length  # Number of patches (reduced from 256)
        self.d_model = self.input_dims  # Model dimension
        self.num_heads = num_heads  # Increased number of attention heads (works well with 256 dim)
        self.dff = dff  # Feed-forward network dimension
        self.dropout = dropout  # Dropout rate

        # Calculate key dimension (must be divisible by num_heads)
        self.d_k = self.d_model // self.num_heads

        # Query, key, and value projections for attention
        self.W_q = nn.Linear(self.d_model, self.d_k * self.num_heads)
        self.W_k = nn.Linear(self.d_model, self.d_k * self.num_heads)
        self.W_v = nn.Linear(self.d_model, self.d_k * self.num_heads)

        # Context projections for hidden states
        self.W_C1q = nn.Linear(self.dff * 4, self.d_k * self.num_heads)
        self.W_C1k = nn.Linear(self.dff * 4, self.d_k * self.num_heads)
        self.W_C1v = nn.Linear(self.dff * 4, self.d_k * self.num_heads)

        # Layer normalization for attention components
        self.normQE = nn.LayerNorm(self.d_model)
        self.normKE = nn.LayerNorm(self.d_model)
        self.normVE = nn.LayerNorm(self.d_model)
        self.normQH = nn.LayerNorm(self.d_model)
        self.normKH = nn.LayerNorm(self.d_model)
        self.normVH = nn.LayerNorm(self.d_model)

        # Feed-forward network layers
        self.linear1 = nn.Linear(self.num_heads * self.d_k, self.d_model * 4)
        self.linear2 = nn.Linear(self.d_model * 4, self.d_model)
        self.dropout1 = nn.Dropout(self.dropout)

        # Normalization and dropout for feed-forward path
        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)
        self.dropout2 = nn.Dropout(self.dropout)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, C1, C2, C3, C4):
        """
        Forward pass through the transformer block.

        Args:
            state: Input state tensor
            H: Hidden state from LSTM

        Returns:
            Z5: Processed state tensor
        """
        # Reshape input state
        # print('ayo',state.shape)

        state = state.view(-1, self.patch_length, self.input_dims)
        # print('ayo',state.shape)
        # Reshape hidden state for context
        batch_size = state.shape[0]

        C1 = C1.view(-1, self.patch_length, self.dff)
        C2 = C2.view(-1, self.patch_length, self.dff)
        C3 = C3.view(-1, self.patch_length, self.dff)
        C4 = C4.view(-1, self.patch_length, self.dff)
        C = torch.cat((C1, C2, C3, C4), dim=-1)
        # Attention mechanism
        # Construct query, key, and value with both input state and hidden state context
        q = self.normQE(self.W_q(state)).view(batch_size, -1, self.num_heads, self.d_k) * self.normQH(
            self.W_C1q(C)).view(batch_size, -1, self.num_heads, self.d_k)
        k = self.normKE(self.W_k(state)).view(batch_size, -1, self.num_heads, self.d_k) * self.normKH(
            self.W_C1k(C)).view(batch_size, -1, self.num_heads, self.d_k)
        v = self.normVE(self.W_v(state)).view(batch_size, -1, self.num_heads, self.d_k) * self.normVH(
            self.W_C1v(C)).view(batch_size, -1, self.num_heads, self.d_k)

        # Reshape for multi-head attention
        q, k, v = [x.transpose(1, 2) for x in (q, k, v)]  # [batch_size, num_heads, seq_len, d_k]

        # Compute attention and get values
        attn_values, attn_weights = self.calculate_attention(q, k, v)
        attn_values = attn_values.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)

        # Residual connection and dropout
        Z1 = state + self.dropout1(attn_values)

        # Feed-forward network with residual connection
        Z2_norm = self.norm1(Z1)
        Z3 = F.gelu(self.linear1(Z2_norm),approximate='tanh')
        Z4 = self.linear2(Z3)
        dZ = self.dropout1(attn_values) + self.dropout2(Z4)
        Z5 = state + dZ

        return Z5, dZ, state

    # RVITS.py, in TransformerBlockX.calculate_attention
    def calculate_attention(self, q, k, v):
        """
        Calculate scaled dot-product attention using fused kernels (SDPA) for speedup.
        """
        # NO NEED FOR MANUAL SCALING OR SOFTMAX

        # Use fused SDPA. SDPA automatically handles the scaling by sqrt(d_k)
        # We assume no attention mask is needed here (is_causal=False, attn_mask=None)
        attention_output = F.scaled_dot_product_attention(
            query=q, 
            key=k, 
            value=v, 
            attn_mask=None, 
            dropout_p=0.0, # Dropout should be applied *after* this fusion 
                        # but can be added here if needed; setting to 0.0 for now
            is_causal=False
        )

        # NOTE: SDPA does not return the weights (attention_weights). 
        # If you require the weights for interpretability/logging, you must keep 
        # the original code. For pure speedup, this is the best path.

        # For compatibility with your current function signature:
        # Since SDPA doesn't return weights, we return None for attention_weights 
        # as it is unused in the TransformerBlockX.forward methods.
        attention_weights = None 

        return attention_output, attention_weights

class TransformerBlockM5(nn.Module):
    """
    Transformer block that processes input state with self-attention mechanism.
    Incorporates hidden states from LSTMs for additional context.
    """

    def __init__(self, input_dims=64, patch_length=12 * 16 + 1, dff=256, dropout=0.1, num_heads=2):
        super(TransformerBlockM5, self).__init__()
        self.input_dims = input_dims  # Using reduced dimension from CNN (256)
        self.patch_length = patch_length  # Number of patches (reduced from 256)
        self.d_model = self.input_dims  # Model dimension
        self.num_heads = num_heads  # Increased number of attention heads (works well with 256 dim)
        self.dff = dff  # Feed-forward network dimension
        self.dropout = dropout  # Dropout rate

        # Calculate key dimension (must be divisible by num_heads)
        self.d_k = self.d_model // self.num_heads

        # Query, key, and value projections for attention
        self.W_q = nn.Linear(self.d_model, self.d_k * self.num_heads)
        self.W_k = nn.Linear(self.d_model, self.d_k * self.num_heads)
        self.W_v = nn.Linear(self.d_model, self.d_k * self.num_heads)

        # Context projections for hidden states
        self.W_C1q = nn.Linear(self.dff * 5, self.d_k * self.num_heads)
        self.W_C1k = nn.Linear(self.dff * 5, self.d_k * self.num_heads)
        self.W_C1v = nn.Linear(self.dff * 5, self.d_k * self.num_heads)

        # Layer normalization for attention components
        self.normQE = nn.LayerNorm(self.d_model)
        self.normKE = nn.LayerNorm(self.d_model)
        self.normVE = nn.LayerNorm(self.d_model)
        self.normQH = nn.LayerNorm(self.d_model)
        self.normKH = nn.LayerNorm(self.d_model)
        self.normVH = nn.LayerNorm(self.d_model)

        # Feed-forward network layers
        self.linear1 = nn.Linear(self.num_heads * self.d_k, self.d_model * 4)
        self.linear2 = nn.Linear(self.d_model * 4, self.d_model)
        self.dropout1 = nn.Dropout(self.dropout)

        # Normalization and dropout for feed-forward path
        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)
        self.dropout2 = nn.Dropout(self.dropout)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, C1, C2, C3, C4, C5):
        """
        Forward pass through the transformer block.

        Args:
            state: Input state tensor
            H: Hidden state from LSTM

        Returns:
            Z5: Processed state tensor
        """
        # Reshape input state
        # print('ayo',state.shape)

        state = state.view(-1, self.patch_length, self.input_dims)
        # print('ayo',state.shape)
        # Reshape hidden state for context
        batch_size = state.shape[0]

        C1 = C1.view(-1, self.patch_length, self.dff)
        C2 = C2.view(-1, self.patch_length, self.dff)
        C3 = C3.view(-1, self.patch_length, self.dff)
        C4 = C4.view(-1, self.patch_length, self.dff)
        C5 = C5.view(-1, self.patch_length, self.dff)
        C = torch.cat((C1, C2, C3, C4, C5), dim=-1)
        # Attention mechanism
        # Construct query, key, and value with both input state and hidden state context
        q = self.normQE(self.W_q(state)).view(batch_size, -1, self.num_heads, self.d_k) * self.normQH(
            self.W_C1q(C)).view(batch_size, -1, self.num_heads, self.d_k)
        k = self.normKE(self.W_k(state)).view(batch_size, -1, self.num_heads, self.d_k) * self.normKH(
            self.W_C1k(C)).view(batch_size, -1, self.num_heads, self.d_k)
        v = self.normVE(self.W_v(state)).view(batch_size, -1, self.num_heads, self.d_k) * self.normVH(
            self.W_C1v(C)).view(batch_size, -1, self.num_heads, self.d_k)

        # Reshape for multi-head attention
        q, k, v = [x.transpose(1, 2) for x in (q, k, v)]  # [batch_size, num_heads, seq_len, d_k]

        # Compute attention and get values
        attn_values, attn_weights = self.calculate_attention(q, k, v)
        attn_values = attn_values.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)

        # Residual connection and dropout
        Z1 = state + self.dropout1(attn_values)

        # Feed-forward network with residual connection
        Z2_norm = self.norm1(Z1)
        Z3 = F.gelu(self.linear1(Z2_norm),approximate='tanh')
        Z4 = self.linear2(Z3)
        dZ = self.dropout1(attn_values) + self.dropout2(Z4)
        Z5 = state + dZ

        return Z5, dZ, state

    # RVITS.py, in TransformerBlockX.calculate_attention
    def calculate_attention(self, q, k, v):
        """
        Calculate scaled dot-product attention using fused kernels (SDPA) for speedup.
        """
        # NO NEED FOR MANUAL SCALING OR SOFTMAX

        # Use fused SDPA. SDPA automatically handles the scaling by sqrt(d_k)
        # We assume no attention mask is needed here (is_causal=False, attn_mask=None)
        attention_output = F.scaled_dot_product_attention(
            query=q, 
            key=k, 
            value=v, 
            attn_mask=None, 
            dropout_p=0.0, # Dropout should be applied *after* this fusion 
                        # but can be added here if needed; setting to 0.0 for now
            is_causal=False
        )

        # NOTE: SDPA does not return the weights (attention_weights). 
        # If you require the weights for interpretability/logging, you must keep 
        # the original code. For pure speedup, this is the best path.

        # For compatibility with your current function signature:
        # Since SDPA doesn't return weights, we return None for attention_weights 
        # as it is unused in the TransformerBlockX.forward methods.
        attention_weights = None 

        return attention_output, attention_weights



class CustomLSTMCell0(nn.Module):
    def __init__(self, patch_size=12 * 16, d_model=64, dff=256, noise_std=0.005):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        self.dff = dff
        self.noise_std = noise_std

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

        self.TM1 = TransformerBlockM0(input_dims=dff, patch_length=patch_size, dff=dff, dropout=0.1, num_heads=32)
        self.normC1 = nn.LayerNorm(dff)
        self.normC2 = nn.LayerNorm(dff)

        # New VAE projection heads

    def _lstm_step(self, Zi, Ci):
        """Internal function to compute a single LSTM step, used for checkpointing"""
        # reshape
        Zi = Zi.view(-1, self.patch_size, self.d_model)
        Ci = Ci.view(-1, self.patch_size, self.dff)

        # add noise
        noise_C1 = torch.randn_like(Ci) * self.noise_std
        noise_C2 = torch.randn_like(Ci) * self.noise_std
        noise_Z1 = torch.randn_like(Zi) * self.noise_std
        noise_Z2 = torch.randn_like(Zi) * self.noise_std
        C_noisy1 = Ci + noise_C1
        C_noisy2 = Ci + noise_C2
        Z_noisy1 = Zi + noise_Z1
        Z_noisy2 = Zi + noise_Z2

        # gate pre-activations
        I_tilde = self.WI1(C_noisy1) + self.RI1(Z_noisy1) + self.WI2(C_noisy2) + self.RI2(Z_noisy2)
        F_tilde = self.WF1(C_noisy1) + self.RF1(Z_noisy1) + self.WF2(C_noisy2) + self.RF2(Z_noisy2)
        Z_tilde = self.WZ1(C_noisy1) + self.RZ1(Z_noisy1) + self.WZ2(C_noisy2) + self.RZ2(Z_noisy2)

        # numerically stable gating
        I_tilde = F.gelu(I_tilde)
        F_tilde = F.gelu(F_tilde)
        I_t = T.exp(-I_tilde)
        F_t = T.exp(-F_tilde)

        # activations
        Z_t = Z_tilde

        # cell update
        C_t = self.normC1(Ci * F_t + Z_t * I_t)
        C_t_ = self.TM1(C_t)[0]

        noise_C1 = C_t + torch.randn_like(C_t) * self.noise_std
        noise_C2 = C_t + torch.randn_like(C_t) * self.noise_std
        noise_C1_ = C_t_ + torch.randn_like(C_t_) * self.noise_std
        noise_C2_ = C_t_ + torch.randn_like(C_t_) * self.noise_std
        O_tilde = self.WO1(noise_C1) + self.RO1(noise_C1_) + self.WO2(noise_C2) + self.RO2(noise_C2_)
        O_t = F.sigmoid(O_tilde.clamp(min=-20, max=20))

        C_t = self.normC2(O_t * C_t + (1 - O_t) * C_t_)
        return C_t

    def forward(self, Zi, Ci):
        # Use checkpointing for LSTM computation to save memory
        return self._lstm_step(Zi, Ci)


class CustomLSTMCell1(nn.Module):
    def __init__(self, patch_size=12 * 16, d_model=64, dff=256, noise_std=0.05):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        self.dff = dff
        self.noise_std = noise_std

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

        self.TM1 = TransformerBlockM1(input_dims=dff, patch_length=patch_size, dff=dff, dropout=0.1, num_heads=16)
        self.normC1 = nn.LayerNorm(dff)
        self.normC2 = nn.LayerNorm(dff)

        # New VAE projection heads

    def _lstm_step(self, Zi, Ci, C_):
        """Internal function to compute a single LSTM step, used for checkpointing"""
        # reshape
        Zi = Zi.view(-1, self.patch_size, self.d_model)
        Ci = Ci.view(-1, self.patch_size, self.dff)

        # add noise
        noise_C1 = torch.randn_like(Ci) * self.noise_std
        noise_C2 = torch.randn_like(Ci) * self.noise_std
        noise_Z1 = torch.randn_like(Zi) * self.noise_std
        noise_Z2 = torch.randn_like(Zi) * self.noise_std
        C_noisy1 = Ci + noise_C1
        C_noisy2 = Ci + noise_C2
        Z_noisy1 = Zi + noise_Z1
        Z_noisy2 = Zi + noise_Z2

        # gate pre-activations
        I_tilde = self.WI1(C_noisy1) + self.RI1(Z_noisy1) + self.WI2(C_noisy2) + self.RI2(Z_noisy2)
        F_tilde = self.WF1(C_noisy1) + self.RF1(Z_noisy1) + self.WF2(C_noisy2) + self.RF2(Z_noisy2)
        Z_tilde = self.WZ1(C_noisy1) + self.RZ1(Z_noisy1) + self.WZ2(C_noisy2) + self.RZ2(Z_noisy2)

        # numerically stable gating
        I_tilde = F.gelu(I_tilde)
        F_tilde = F.gelu(F_tilde)
        I_t = T.exp(-I_tilde)
        F_t = T.exp(-F_tilde)

        # activations
        Z_t = Z_tilde

        # cell update
        C_t = self.normC1(Ci * F_t + Z_t * I_t)
        C_t_ = self.TM1(C_t, C_)[0]

        noise_C1 = C_t + torch.randn_like(C_t) * self.noise_std
        noise_C2 = C_t + torch.randn_like(C_t) * self.noise_std
        noise_C1_ = C_t_ + torch.randn_like(C_t_) * self.noise_std
        noise_C2_ = C_t_ + torch.randn_like(C_t_) * self.noise_std
        O_tilde = self.WO1(noise_C1) + self.RO1(noise_C1_) + self.WO2(noise_C2) + self.RO2(noise_C2_)
        O_t = F.sigmoid(O_tilde.clamp(min=-20, max=20))

        C_t = self.normC2(O_t * C_t + (1 - O_t) * C_t_)

        return C_t

    def forward(self, Zi, Ci, C_):
        # Use checkpointing for LSTM computation to save memory
        return self._lstm_step(Zi, Ci, C_)

class CustomLSTMCell2(nn.Module):
    def __init__(self, patch_size=12 * 16, d_model=64, dff=256, noise_std=0.005):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        self.dff = dff
        self.noise_std = noise_std

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

        self.TM1 = TransformerBlockM2(input_dims=dff, patch_length=patch_size, dff=dff, dropout=0.1, num_heads=8)
        self.normC1 = nn.LayerNorm(dff)
        self.normC2 = nn.LayerNorm(dff)

        # New VAE projection heads

    def _lstm_step(self, Zi, Ci, C1_, C2_):
        """Internal function to compute a single LSTM step, used for checkpointing"""
        # reshape
        Zi = Zi.view(-1, self.patch_size, self.d_model)
        Ci = Ci.view(-1, self.patch_size, self.dff)

        # add noise
        noise_C1 = torch.randn_like(Ci) * self.noise_std
        noise_C2 = torch.randn_like(Ci) * self.noise_std
        noise_Z1 = torch.randn_like(Zi) * self.noise_std
        noise_Z2 = torch.randn_like(Zi) * self.noise_std
        C_noisy1 = Ci + noise_C1
        C_noisy2 = Ci + noise_C2
        Z_noisy1 = Zi + noise_Z1
        Z_noisy2 = Zi + noise_Z2

        # gate pre-activations
        I_tilde = self.WI1(C_noisy1) + self.RI1(Z_noisy1) + self.WI2(C_noisy2) + self.RI2(Z_noisy2)
        F_tilde = self.WF1(C_noisy1) + self.RF1(Z_noisy1) + self.WF2(C_noisy2) + self.RF2(Z_noisy2)
        Z_tilde = self.WZ1(C_noisy1) + self.RZ1(Z_noisy1) + self.WZ2(C_noisy2) + self.RZ2(Z_noisy2)

        # numerically stable gating
        I_tilde = F.gelu(I_tilde)
        F_tilde = F.gelu(F_tilde)
        I_t = T.exp(-I_tilde)
        F_t = T.exp(-F_tilde)

        # activations
        Z_t = Z_tilde

        # cell update
        C_t = self.normC1(Ci * F_t + Z_t * I_t)
        # C_t = C_t/(torch.max(torch.abs(C_t),dim=1,keepdim=True)[0]+1e-8)
        # print('C_t', torch.max(C_t))
        C_t_ = self.TM1(C_t, C1_, C2_)[0]

        noise_C1 = C_t + torch.randn_like(C_t) * self.noise_std
        noise_C2 = C_t + torch.randn_like(C_t) * self.noise_std
        noise_C1_ = C_t_ + torch.randn_like(C_t_) * self.noise_std
        noise_C2_ = C_t_ + torch.randn_like(C_t_) * self.noise_std
        O_tilde = self.WO1(noise_C1) + self.RO1(noise_C1_) + self.WO2(noise_C2) + self.RO2(noise_C2_)
        O_t = F.sigmoid(O_tilde.clamp(min=-20, max=20))

        C_t = self.normC2(O_t * C_t + (1 - O_t) * C_t_)

        return C_t

    def forward(self, Zi, Ci, C1_, C2_):
        # Use checkpointing for LSTM computation to save memory
        return self._lstm_step(Zi, Ci, C1_, C2_)


class CustomLSTMCell3(nn.Module):
    def __init__(self, patch_size=12 * 16, d_model=64, dff=256, noise_std=0.005):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        self.dff = dff
        self.noise_std = noise_std

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

        self.TM1 = TransformerBlockM3(input_dims=dff, patch_length=patch_size, dff=dff, dropout=0.1, num_heads=8)
        self.normC1 = nn.LayerNorm(dff)
        self.normC2 = nn.LayerNorm(dff)

        # New VAE projection heads

    def _lstm_step(self, Zi, Ci, C1_, C2_, C3_):
        """Internal function to compute a single LSTM step, used for checkpointing"""
        # reshape
        Zi = Zi.view(-1, self.patch_size, self.d_model)
        Ci = Ci.view(-1, self.patch_size, self.dff)

        # add noise
        noise_C1 = torch.randn_like(Ci) * self.noise_std
        noise_C2 = torch.randn_like(Ci) * self.noise_std
        noise_Z1 = torch.randn_like(Zi) * self.noise_std
        noise_Z2 = torch.randn_like(Zi) * self.noise_std
        C_noisy1 = Ci + noise_C1
        C_noisy2 = Ci + noise_C2
        Z_noisy1 = Zi + noise_Z1
        Z_noisy2 = Zi + noise_Z2

        # gate pre-activations
        I_tilde = self.WI1(C_noisy1) + self.RI1(Z_noisy1) + self.WI2(C_noisy2) + self.RI2(Z_noisy2)
        F_tilde = self.WF1(C_noisy1) + self.RF1(Z_noisy1) + self.WF2(C_noisy2) + self.RF2(Z_noisy2)
        Z_tilde = self.WZ1(C_noisy1) + self.RZ1(Z_noisy1) + self.WZ2(C_noisy2) + self.RZ2(Z_noisy2)

        # numerically stable gating
        I_tilde = F.gelu(I_tilde)
        F_tilde = F.gelu(F_tilde)
        I_t = T.exp(-I_tilde)
        F_t = T.exp(-F_tilde)

        # activations
        Z_t = Z_tilde

        # cell update
        C_t = self.normC1(Ci * F_t + Z_t * I_t)
        # C_t = C_t/(torch.max(torch.abs(C_t),dim=1,keepdim=True)[0]+1e-8)
        # print('C_t', torch.max(C_t))
        C_t_ = self.TM1(C_t, C1_, C2_, C3_)[0]

        noise_C1 = C_t + torch.randn_like(C_t) * self.noise_std
        noise_C2 = C_t + torch.randn_like(C_t) * self.noise_std
        noise_C1_ = C_t_ + torch.randn_like(C_t_) * self.noise_std
        noise_C2_ = C_t_ + torch.randn_like(C_t_) * self.noise_std
        O_tilde = self.WO1(noise_C1) + self.RO1(noise_C1_) + self.WO2(noise_C2) + self.RO2(noise_C2_)
        O_t = F.sigmoid(O_tilde.clamp(min=-20, max=20))

        C_t = self.normC2(O_t * C_t + (1 - O_t) * C_t_)

        return C_t

    def forward(self, Zi, Ci, C1_, C2_, C3_):
        # Use checkpointing for LSTM computation to save memory
        return self._lstm_step(Zi, Ci, C1_, C2_, C3_)

class CustomLSTMCell4(nn.Module):
    def __init__(self, patch_size=12 * 16, d_model=64, dff=256, noise_std=0.005):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        self.dff = dff
        self.noise_std = noise_std

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

        self.TM1 = TransformerBlockM4(input_dims=dff, patch_length=patch_size, dff=dff, dropout=0.1, num_heads=4)
        self.normC1 = nn.LayerNorm(dff)
        self.normC2 = nn.LayerNorm(dff)

        # New VAE projection heads

    def _lstm_step(self, Zi, Ci, C1_, C2_, C3_, C4_):
        """Internal function to compute a single LSTM step, used for checkpointing"""
        # reshape
        Zi = Zi.view(-1, self.patch_size, self.d_model)
        Ci = Ci.view(-1, self.patch_size, self.dff)

        # add noise
        noise_C1 = torch.randn_like(Ci) * self.noise_std
        noise_C2 = torch.randn_like(Ci) * self.noise_std
        noise_Z1 = torch.randn_like(Zi) * self.noise_std
        noise_Z2 = torch.randn_like(Zi) * self.noise_std
        C_noisy1 = Ci + noise_C1
        C_noisy2 = Ci + noise_C2
        Z_noisy1 = Zi + noise_Z1
        Z_noisy2 = Zi + noise_Z2

        # gate pre-activations
        I_tilde = self.WI1(C_noisy1) + self.RI1(Z_noisy1) + self.WI2(C_noisy2) + self.RI2(Z_noisy2)
        F_tilde = self.WF1(C_noisy1) + self.RF1(Z_noisy1) + self.WF2(C_noisy2) + self.RF2(Z_noisy2)
        Z_tilde = self.WZ1(C_noisy1) + self.RZ1(Z_noisy1) + self.WZ2(C_noisy2) + self.RZ2(Z_noisy2)

        # numerically stable gating
        I_tilde = F.gelu(I_tilde)
        F_tilde = F.gelu(F_tilde)
        I_t = T.exp(-I_tilde)
        F_t = T.exp(-F_tilde)

        # activations
        Z_t = Z_tilde

        # cell update
        C_t = self.normC1(Ci * F_t + Z_t * I_t)
        C_t_ = self.TM1(C_t, C1_, C2_, C3_, C4_)[0]

        noise_C1 = C_t + torch.randn_like(C_t) * self.noise_std
        noise_C2 = C_t + torch.randn_like(C_t) * self.noise_std
        noise_C1_ = C_t_ + torch.randn_like(C_t_) * self.noise_std
        noise_C2_ = C_t_ + torch.randn_like(C_t_) * self.noise_std
        O_tilde = self.WO1(noise_C1) + self.RO1(noise_C1_) + self.WO2(noise_C2) + self.RO2(noise_C2_)
        O_t = F.sigmoid(O_tilde.clamp(min=-20, max=20))

        C_t = self.normC2(O_t * C_t + (1 - O_t) * C_t_)


        return C_t

    def forward(self, Zi, Ci, C1_, C2_, C3_, C4_):
        # Use checkpointing for LSTM computation to save memory
        return self._lstm_step(Zi, Ci, C1_, C2_, C3_, C4_)


class CustomLSTMCell5(nn.Module):
    def __init__(self, patch_size=12 * 16, d_model=64, dff=256, noise_std=0.01):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        self.dff = dff
        self.noise_std = noise_std

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

        self.TM1 = TransformerBlockM5(input_dims=dff, patch_length=patch_size, dff=dff, dropout=0.1, num_heads=4)
        self.normC1 = nn.LayerNorm(dff)
        self.normC2 = nn.LayerNorm(dff)

        # New VAE projection heads

    def _lstm_step(self, Zi, Ci, C1_, C2_, C3_, C4_, C5_):
        """Internal function to compute a single LSTM step, used for checkpointing"""
        # reshape
        Zi = Zi.view(-1, self.patch_size, self.d_model)
        Ci = Ci.view(-1, self.patch_size, self.dff)

        # add noise
        noise_C1 = torch.randn_like(Ci) * self.noise_std
        noise_C2 = torch.randn_like(Ci) * self.noise_std
        noise_Z1 = torch.randn_like(Zi) * self.noise_std
        noise_Z2 = torch.randn_like(Zi) * self.noise_std
        C_noisy1 = Ci + noise_C1
        C_noisy2 = Ci + noise_C2
        Z_noisy1 = Zi + noise_Z1
        Z_noisy2 = Zi + noise_Z2

        # gate pre-activations
        I_tilde = self.WI1(C_noisy1) + self.RI1(Z_noisy1) + self.WI2(C_noisy2) + self.RI2(Z_noisy2)
        F_tilde = self.WF1(C_noisy1) + self.RF1(Z_noisy1) + self.WF2(C_noisy2) + self.RF2(Z_noisy2)
        Z_tilde = self.WZ1(C_noisy1) + self.RZ1(Z_noisy1) + self.WZ2(C_noisy2) + self.RZ2(Z_noisy2)

        # numerically stable gating
        I_tilde = F.gelu(I_tilde)
        F_tilde = F.gelu(F_tilde)
        I_t = T.exp(-I_tilde)
        F_t = T.exp(-F_tilde)

        # activations
        Z_t = Z_tilde

        # cell update
        C_t = self.normC1(Ci * F_t + Z_t * I_t)
        C_t_ = self.TM1(C_t, C1_, C2_, C3_, C4_, C5_)[0]

        noise_C1 = C_t + torch.randn_like(C_t) * self.noise_std
        noise_C2 = C_t + torch.randn_like(C_t) * self.noise_std
        noise_C1_ = C_t_ + torch.randn_like(C_t_) * self.noise_std
        noise_C2_ = C_t_ + torch.randn_like(C_t_) * self.noise_std
        O_tilde = self.WO1(noise_C1) + self.RO1(noise_C1_) + self.WO2(noise_C2) + self.RO2(noise_C2_)
        O_t = F.sigmoid(O_tilde.clamp(min=-20, max=20))

        C_t = self.normC2(O_t * C_t + (1 - O_t) * C_t_)


        return C_t

    def forward(self, Zi, Ci, C1_, C2_, C3_, C4_, C5_):
        # Use checkpointing for LSTM computation to save memory
        return self._lstm_step(Zi, Ci, C1_, C2_, C3_, C4_, C5_)

class TransformerBlock4(nn.Module):
    """
    Transformer block that processes input state with self-attention mechanism.
    Incorporates hidden states from LSTMs for additional context.
    """

    def __init__(self, input_dims=64, patch_length=12 * 16 + 1, dff=256, dropout=0.1, num_heads=2):
        super(TransformerBlock4, self).__init__()
        self.input_dims = input_dims  # Using reduced dimension from CNN (256)
        self.patch_length = patch_length  # Number of patches (reduced from 256)
        self.d_model = self.input_dims  # Model dimension
        self.num_heads = num_heads  # Increased number of attention heads (works well with 256 dim)
        self.dff = dff  # Feed-forward network dimension
        self.dropout = dropout  # Dropout rate

        # Calculate key dimension (must be divisible by num_heads)
        self.d_k = self.d_model // self.num_heads

        # Query, key, and value projections for attention
        self.W_q = nn.Linear(self.d_model, self.d_k * self.num_heads)
        self.W_k = nn.Linear(self.d_model, self.d_k * self.num_heads)
        self.W_v = nn.Linear(self.d_model, self.d_k * self.num_heads)

        # Context projections for hidden states
        self.W_C1q = nn.Linear(self.dff * 4, self.d_k * self.num_heads)
        self.W_C1k = nn.Linear(self.dff * 4, self.d_k * self.num_heads)
        self.W_C1v = nn.Linear(self.dff * 4, self.d_k * self.num_heads)

        # Layer normalization for attention components
        self.normQE = nn.LayerNorm(self.d_model)
        self.normKE = nn.LayerNorm(self.d_model)
        self.normVE = nn.LayerNorm(self.d_model)
        self.normQH = nn.LayerNorm(self.d_model)
        self.normKH = nn.LayerNorm(self.d_model)
        self.normVH = nn.LayerNorm(self.d_model)

        # Feed-forward network layers
        self.linear1 = nn.Linear(self.num_heads * self.d_k, self.d_model * 4)
        self.linear2 = nn.Linear(self.d_model * 4, self.d_model)
        self.dropout1 = nn.Dropout(self.dropout)

        # Normalization and dropout for feed-forward path
        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)
        self.dropout2 = nn.Dropout(self.dropout)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, C1, C2, C3, C4):
        """
        Forward pass through the transformer block.

        Args:
            state: Input state tensor
            H: Hidden state from LSTM

        Returns:
            Z5: Processed state tensor
        """
        # Reshape input state
        state = state.view(-1, self.patch_length, self.input_dims)
        # state = state.view(-1, self.patch_length, self.input_dims)
        C1 = C1.view(-1, self.patch_length, self.dff)
        C2 = C2.view(-1, self.patch_length, self.dff)
        C3 = C3.view(-1, self.patch_length, self.dff)
        C4 = C4.view(-1, self.patch_length, self.dff)
        # concatenate along the last axis to get (B, patch_length, 2*dff)
        C_cat = torch.cat([C1, C2, C3, C4], dim=-1)
        batch_size = state.shape[0]

        # Compute queries, keys, and values
        q = self.normQE(self.W_q(state)).view(batch_size, -1, self.num_heads, self.d_k) * self.normQH(
            self.W_C1q(C_cat)).view(batch_size, -1, self.num_heads, self.d_k)
        k = self.normKE(self.W_k(state)).view(batch_size, -1, self.num_heads, self.d_k) * self.normKH(
            self.W_C1k(C_cat)).view(batch_size, -1, self.num_heads, self.d_k)
        v = self.normVE(self.W_v(state)).view(batch_size, -1, self.num_heads, self.d_k) * self.normVH(
            self.W_C1v(C_cat)).view(batch_size, -1, self.num_heads, self.d_k)

        # Reshape hidden state for context
        # print('H',H.shape)
        # H = H.view(-1, self.patch_length, self.dff)
        batch_size = state.shape[0]


        # Reshape for multi-head attention
        q, k, v = [x.transpose(1, 2) for x in (q, k, v)]  # [batch_size, num_heads, seq_len, d_k]

        # Compute attention and get values
        attn_values, attn_weights = self.calculate_attention(q, k, v)
        attn_values = attn_values.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)

        # Residual connection and dropout
        Z1 = state + self.dropout1(attn_values)

        # Feed-forward network with residual connection
        Z2_norm = self.norm1(Z1)
        Z3 = F.gelu(self.linear1(Z2_norm))
        Z4 = self.linear2(Z3)
        dZ = self.dropout1(attn_values) + self.dropout2(Z4)
        Z5 = state + dZ

        return Z5

    def calculate_attention(self, q, k, v):
        """
        Calculate scaled dot-product attention.

        Args:
            q: Query tensor
            k: Key tensor
            v: Value tensor

        Returns:
            attention_output: Output after applying attention to values
            attention_weights: The attention weights
        """
        # Compute attention scores with scaling
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))

        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)

        # Compute attention output
        attention_output = torch.matmul(attention_weights, v)

        return attention_output, attention_weights


class TransformerBlock5(nn.Module):
    """
    Transformer block that processes input state with self-attention mechanism.
    Incorporates hidden states from LSTMs for additional context.
    """

    def __init__(self, input_dims=64, patch_length=12 * 16 + 1, dff=256, dropout=0.1, num_heads=2):
        super(TransformerBlock5, self).__init__()
        self.input_dims = input_dims  # Using reduced dimension from CNN (256)
        self.patch_length = patch_length  # Number of patches (reduced from 256)
        self.d_model = self.input_dims  # Model dimension
        self.num_heads = num_heads  # Increased number of attention heads (works well with 256 dim)
        self.dff = dff  # Feed-forward network dimension
        self.dropout = dropout  # Dropout rate

        # Calculate key dimension (must be divisible by num_heads)
        self.d_k = self.d_model // self.num_heads

        # Query, key, and value projections for attention
        self.W_q = nn.Linear(self.d_model, self.d_k * self.num_heads)
        self.W_k = nn.Linear(self.d_model, self.d_k * self.num_heads)
        self.W_v = nn.Linear(self.d_model, self.d_k * self.num_heads)

        # Context projections for hidden states
        self.W_C1q = nn.Linear(self.dff * 5, self.d_k * self.num_heads)
        self.W_C1k = nn.Linear(self.dff * 5, self.d_k * self.num_heads)
        self.W_C1v = nn.Linear(self.dff * 5, self.d_k * self.num_heads)

        # Layer normalization for attention components
        self.normQE = nn.LayerNorm(self.d_model)
        self.normKE = nn.LayerNorm(self.d_model)
        self.normVE = nn.LayerNorm(self.d_model)
        self.normQH = nn.LayerNorm(self.d_model)
        self.normKH = nn.LayerNorm(self.d_model)
        self.normVH = nn.LayerNorm(self.d_model)

        # Feed-forward network layers
        self.linear1 = nn.Linear(self.num_heads * self.d_k, self.d_model * 4)
        self.linear2 = nn.Linear(self.d_model * 4, self.d_model)
        self.dropout1 = nn.Dropout(self.dropout)

        # Normalization and dropout for feed-forward path
        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)
        self.dropout2 = nn.Dropout(self.dropout)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, C1, C2, C3, C4, C5):
        """
        Forward pass through the transformer block.

        Args:
            state: Input state tensor
            H: Hidden state from LSTM

        Returns:
            Z5: Processed state tensor
        """
        # Reshape input state
        state = state.view(-1, self.patch_length, self.input_dims)
        # state = state.view(-1, self.patch_length, self.input_dims)
        C1 = C1.view(-1, self.patch_length, self.dff)
        C2 = C2.view(-1, self.patch_length, self.dff)
        C3 = C3.view(-1, self.patch_length, self.dff)
        C4 = C4.view(-1, self.patch_length, self.dff)
        C5 = C5.view(-1, self.patch_length, self.dff)
        # concatenate along the last axis to get (B, patch_length, 2*dff)
        C_cat = torch.cat([C1, C2, C3, C4, C5], dim=-1)
        batch_size = state.shape[0]

        # Compute queries, keys, and values
        q = self.normQE(self.W_q(state)).view(batch_size, -1, self.num_heads, self.d_k) * self.normQH(
            self.W_C1q(C_cat)).view(batch_size, -1, self.num_heads, self.d_k)
        k = self.normKE(self.W_k(state)).view(batch_size, -1, self.num_heads, self.d_k) * self.normKH(
            self.W_C1k(C_cat)).view(batch_size, -1, self.num_heads, self.d_k)
        v = self.normVE(self.W_v(state)).view(batch_size, -1, self.num_heads, self.d_k) * self.normVH(
            self.W_C1v(C_cat)).view(batch_size, -1, self.num_heads, self.d_k)

        # Reshape hidden state for context
        # print('H',H.shape)
        # H = H.view(-1, self.patch_length, self.dff)
        batch_size = state.shape[0]


        # Reshape for multi-head attention
        q, k, v = [x.transpose(1, 2) for x in (q, k, v)]  # [batch_size, num_heads, seq_len, d_k]

        # Compute attention and get values
        attn_values, attn_weights = self.calculate_attention(q, k, v)
        attn_values = attn_values.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)

        # Residual connection and dropout
        Z1 = state + self.dropout1(attn_values)

        # Feed-forward network with residual connection
        Z2_norm = self.norm1(Z1)
        Z3 = F.gelu(self.linear1(Z2_norm))
        Z4 = self.linear2(Z3)
        dZ = self.dropout1(attn_values) + self.dropout2(Z4)
        Z5 = state + dZ

        return Z5

    def calculate_attention(self, q, k, v):
        """
        Calculate scaled dot-product attention.

        Args:
            q: Query tensor
            k: Key tensor
            v: Value tensor

        Returns:
            attention_output: Output after applying attention to values
            attention_weights: The attention weights
        """
        # Compute attention scores with scaling
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))

        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)

        # Compute attention output
        attention_output = torch.matmul(attention_weights, v)

        return attention_output, attention_weights



class TransformerBlock8(nn.Module):
    """
    Transformer block that processes input state with self-attention mechanism.
    Incorporates hidden states from LSTMs for additional context.
    """

    def __init__(self, input_dims=64, patch_length=12 * 16 + 1, dff=256, dropout=0.1, num_heads=2):
        super(TransformerBlock8, self).__init__()
        self.input_dims = input_dims  # Using reduced dimension from CNN (256)
        self.patch_length = patch_length  # Number of patches (reduced from 256)
        self.d_model = self.input_dims  # Model dimension
        self.num_heads = num_heads  # Increased number of attention heads (works well with 256 dim)
        self.dff = dff  # Feed-forward network dimension
        self.dropout = dropout  # Dropout rate

        # Calculate key dimension (must be divisible by num_heads)
        self.d_k = self.d_model // self.num_heads

        # Query, key, and value projections for attention
        self.W_q = nn.Linear(self.d_model, self.d_k * self.num_heads)
        self.W_k = nn.Linear(self.d_model, self.d_k * self.num_heads)
        self.W_v = nn.Linear(self.d_model, self.d_k * self.num_heads)

        # Context projections for hidden states
        self.W_C1q = nn.Linear(self.dff * 8, self.d_k * self.num_heads)
        self.W_C1k = nn.Linear(self.dff * 8, self.d_k * self.num_heads)
        self.W_C1v = nn.Linear(self.dff * 8, self.d_k * self.num_heads)

        # Layer normalization for attention components
        self.normQE = nn.LayerNorm(self.d_model)
        self.normKE = nn.LayerNorm(self.d_model)
        self.normVE = nn.LayerNorm(self.d_model)
        self.normQH = nn.LayerNorm(self.d_model)
        self.normKH = nn.LayerNorm(self.d_model)
        self.normVH = nn.LayerNorm(self.d_model)

        # Feed-forward network layers
        self.linear1 = nn.Linear(self.num_heads * self.d_k, self.d_model * 4)
        self.linear2 = nn.Linear(self.d_model * 4, self.d_model)
        self.dropout1 = nn.Dropout(self.dropout)

        # Normalization and dropout for feed-forward path
        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)
        self.dropout2 = nn.Dropout(self.dropout)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, C1, C2, C3, C4, C5, C6, C7, C8):
        """
        Forward pass through the transformer block.

        Args:
            state: Input state tensor
            H: Hidden state from LSTM

        Returns:
            Z5: Processed state tensor
        """
        # Reshape input state
        state = state.view(-1, self.patch_length, self.input_dims)
        # state = state.view(-1, self.patch_length, self.input_dims)
        C1 = C1.view(-1, self.patch_length, self.dff)
        C2 = C2.view(-1, self.patch_length, self.dff)
        C3 = C3.view(-1, self.patch_length, self.dff)
        C4 = C4.view(-1, self.patch_length, self.dff)
        C5 = C5.view(-1, self.patch_length, self.dff)
        C6 = C6.view(-1, self.patch_length, self.dff)
        C7 = C7.view(-1, self.patch_length, self.dff)
        C8 = C8.view(-1, self.patch_length, self.dff)
        # concatenate along the last axis to get (B, patch_length, 2*dff)
        C_cat = torch.cat([C1, C2, C3, C4, C5, C6, C7, C8], dim=-1)
        batch_size = state.shape[0]

        # Compute queries, keys, and values
        q = self.normQE(self.W_q(state)).view(batch_size, -1, self.num_heads, self.d_k) * self.normQH(
            self.W_C1q(C_cat)).view(batch_size, -1, self.num_heads, self.d_k)
        k = self.normKE(self.W_k(state)).view(batch_size, -1, self.num_heads, self.d_k) * self.normKH(
            self.W_C1k(C_cat)).view(batch_size, -1, self.num_heads, self.d_k)
        v = self.normVE(self.W_v(state)).view(batch_size, -1, self.num_heads, self.d_k) * self.normVH(
            self.W_C1v(C_cat)).view(batch_size, -1, self.num_heads, self.d_k)

        # Reshape hidden state for context
        # print('H',H.shape)
        # H = H.view(-1, self.patch_length, self.dff)
        batch_size = state.shape[0]


        # Reshape for multi-head attention
        q, k, v = [x.transpose(1, 2) for x in (q, k, v)]  # [batch_size, num_heads, seq_len, d_k]

        # Compute attention and get values
        attn_values, attn_weights = self.calculate_attention(q, k, v)
        attn_values = attn_values.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)

        # Residual connection and dropout
        Z1 = state + self.dropout1(attn_values)

        # Feed-forward network with residual connection
        Z2_norm = self.norm1(Z1)
        Z3 = F.gelu(self.linear1(Z2_norm))
        Z4 = self.linear2(Z3)
        dZ = self.dropout1(attn_values) + self.dropout2(Z4)
        Z5 = state + dZ

        return Z5

    def calculate_attention(self, q, k, v):
        """
        Calculate scaled dot-product attention.

        Args:
            q: Query tensor
            k: Key tensor
            v: Value tensor

        Returns:
            attention_output: Output after applying attention to values
            attention_weights: The attention weights
        """
        # Compute attention scores with scaling
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))

        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)

        # Compute attention output
        attention_output = torch.matmul(attention_weights, v)

        return attention_output, attention_weights
class TransformerBlock6(nn.Module):
    """
    Transformer block that processes input state with self-attention mechanism.
    Incorporates hidden states from LSTMs for additional context.
    """

    def __init__(self, input_dims=64, patch_length=12 * 16 + 1, dff=256, dropout=0.1, num_heads=2):
        super(TransformerBlock6, self).__init__()
        self.input_dims = input_dims  # Using reduced dimension from CNN (256)
        self.patch_length = patch_length  # Number of patches (reduced from 256)
        self.d_model = self.input_dims  # Model dimension
        self.num_heads = num_heads  # Increased number of attention heads (works well with 256 dim)
        self.dff = dff  # Feed-forward network dimension
        self.dropout = dropout  # Dropout rate

        # Calculate key dimension (must be divisible by num_heads)
        self.d_k = self.d_model // self.num_heads

        # Query, key, and value projections for attention
        self.W_q = nn.Linear(self.d_model, self.d_k * self.num_heads)
        self.W_k = nn.Linear(self.d_model, self.d_k * self.num_heads)
        self.W_v = nn.Linear(self.d_model, self.d_k * self.num_heads)

        # Context projections for hidden states
        self.W_C1q = nn.Linear(self.dff * 6, self.d_k * self.num_heads)
        self.W_C1k = nn.Linear(self.dff * 6, self.d_k * self.num_heads)
        self.W_C1v = nn.Linear(self.dff * 6, self.d_k * self.num_heads)

        # Layer normalization for attention components
        self.normQE = nn.LayerNorm(self.d_model)
        self.normKE = nn.LayerNorm(self.d_model)
        self.normVE = nn.LayerNorm(self.d_model)
        self.normQH = nn.LayerNorm(self.d_model)
        self.normKH = nn.LayerNorm(self.d_model)
        self.normVH = nn.LayerNorm(self.d_model)

        # Feed-forward network layers
        self.linear1 = nn.Linear(self.num_heads * self.d_k, self.d_model * 4)
        self.linear2 = nn.Linear(self.d_model * 4, self.d_model)
        self.dropout1 = nn.Dropout(self.dropout)

        # Normalization and dropout for feed-forward path
        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)
        self.dropout2 = nn.Dropout(self.dropout)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, C1, C2, C3, C4, C5, C6):
        """
        Forward pass through the transformer block.

        Args:
            state: Input state tensor
            H: Hidden state from LSTM

        Returns:
            Z5: Processed state tensor
        """
        # Reshape input state
        state = state.view(-1, self.patch_length, self.input_dims)
        # state = state.view(-1, self.patch_length, self.input_dims)
        C1 = C1.view(-1, self.patch_length, self.dff)
        C2 = C2.view(-1, self.patch_length, self.dff)
        C3 = C3.view(-1, self.patch_length, self.dff)
        C4 = C4.view(-1, self.patch_length, self.dff)
        C5 = C5.view(-1, self.patch_length, self.dff)
        C6 = C6.view(-1, self.patch_length, self.dff)
        # concatenate along the last axis to get (B, patch_length, 2*dff)
        C_cat = torch.cat([C1, C2, C3, C4, C5, C6], dim=-1)
        batch_size = state.shape[0]

        # Compute queries, keys, and values
        q = self.normQE(self.W_q(state)).view(batch_size, -1, self.num_heads, self.d_k) * self.normQH(
            self.W_C1q(C_cat)).view(batch_size, -1, self.num_heads, self.d_k)
        k = self.normKE(self.W_k(state)).view(batch_size, -1, self.num_heads, self.d_k) * self.normKH(
            self.W_C1k(C_cat)).view(batch_size, -1, self.num_heads, self.d_k)
        v = self.normVE(self.W_v(state)).view(batch_size, -1, self.num_heads, self.d_k) * self.normVH(
            self.W_C1v(C_cat)).view(batch_size, -1, self.num_heads, self.d_k)

        # Reshape hidden state for context
        # print('H',H.shape)
        # H = H.view(-1, self.patch_length, self.dff)
        batch_size = state.shape[0]

        # Reshape for multi-head attention
        q, k, v = [x.transpose(1, 2) for x in (q, k, v)]  # [batch_size, num_heads, seq_len, d_k]

        # Compute attention and get values
        attn_values, attn_weights = self.calculate_attention(q, k, v)
        attn_values = attn_values.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)

        # Residual connection and dropout
        Z1 = state + self.dropout1(attn_values)

        # Feed-forward network with residual connection
        Z2_norm = self.norm1(Z1)
        Z3 = F.gelu(self.linear1(Z2_norm))
        Z4 = self.linear2(Z3)
        dZ = self.dropout1(attn_values) + self.dropout2(Z4)
        Z5 = state + dZ

        return Z5

    def calculate_attention(self, q, k, v):
        """
        Calculate scaled dot-product attention.

        Args:
            q: Query tensor
            k: Key tensor
            v: Value tensor

        Returns:
            attention_output: Output after applying attention to values
            attention_weights: The attention weights
        """
        # Compute attention scores with scaling
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))

        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)

        # Compute attention output
        attention_output = torch.matmul(attention_weights, v)

        return attention_output, attention_weights


class TransformerBlock12(nn.Module):
    """
    Transformer block that processes input state with self-attention mechanism.
    Incorporates hidden states from LSTMs for additional context.
    """

    def __init__(self, input_dims=64, patch_length=12 * 16 + 1, dff=256, dropout=0.01, num_heads=2):
        super(TransformerBlock12, self).__init__()
        self.input_dims = input_dims  # Using reduced dimension from CNN (256)
        self.patch_length = patch_length  # Number of patches (reduced from 256)
        self.d_model = self.input_dims  # Model dimension
        self.num_heads = num_heads  # Increased number of attention heads (works well with 256 dim)
        self.dff = dff  # Feed-forward network dimension
        self.dropout = dropout  # Dropout rate

        # Calculate key dimension (must be divisible by num_heads)
        self.d_k = self.d_model // self.num_heads

        # Query, key, and value projections for attention
        self.W_q = nn.Linear(self.d_model, self.d_k * self.num_heads)
        self.W_k = nn.Linear(self.d_model, self.d_k * self.num_heads)
        self.W_v = nn.Linear(self.d_model, self.d_k * self.num_heads)

        # Context projections for hidden states
        self.W_C1q = nn.Linear(self.dff * 12, self.d_k * self.num_heads)
        self.W_C1k = nn.Linear(self.dff * 12, self.d_k * self.num_heads)
        self.W_C1v = nn.Linear(self.dff * 12, self.d_k * self.num_heads)

        # Layer normalization for attention components
        self.normQE = nn.LayerNorm(self.d_model)
        self.normKE = nn.LayerNorm(self.d_model)
        self.normVE = nn.LayerNorm(self.d_model)
        self.normQH = nn.LayerNorm(self.d_model)
        self.normKH = nn.LayerNorm(self.d_model)
        self.normVH = nn.LayerNorm(self.d_model)

        # Feed-forward network layers
        self.linear1 = nn.Linear(self.num_heads * self.d_k, self.d_model * 4)
        self.linear2 = nn.Linear(self.d_model * 4, self.d_model)
        self.dropout1 = nn.Dropout(self.dropout)

        # Normalization and dropout for feed-forward path
        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)
        self.dropout2 = nn.Dropout(self.dropout)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12):
        """
        Forward pass through the transformer block.

        Args:
            state: Input state tensor
            H: Hidden state from LSTM

        Returns:
            Z5: Processed state tensor
        """
        # Reshape input state
        state = state.view(-1, self.patch_length, self.input_dims)
        # state = state.view(-1, self.patch_length, self.input_dims)
        C1 = C1.view(-1, self.patch_length, self.dff)
        C2 = C2.view(-1, self.patch_length, self.dff)
        C3 = C3.view(-1, self.patch_length, self.dff)
        C4 = C4.view(-1, self.patch_length, self.dff)
        C5 = C5.view(-1, self.patch_length, self.dff)
        C6 = C6.view(-1, self.patch_length, self.dff)
        C7 = C7.view(-1, self.patch_length, self.dff)
        C8 = C8.view(-1, self.patch_length, self.dff)
        C9 = C9.view(-1, self.patch_length, self.dff)
        C10 = C10.view(-1, self.patch_length, self.dff)
        C11 = C11.view(-1, self.patch_length, self.dff)
        C12 = C12.view(-1, self.patch_length, self.dff)
        # concatenate along the last axis to get (B, patch_length, 2*dff)
        C_cat = torch.cat([C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12], dim=-1)
        batch_size = state.shape[0]

        # Compute queries, keys, and values
        q = self.normQE(self.W_q(state)).view(batch_size, -1, self.num_heads, self.d_k) * self.normQH(
            self.W_C1q(C_cat)).view(batch_size, -1, self.num_heads, self.d_k)
        k = self.normKE(self.W_k(state)).view(batch_size, -1, self.num_heads, self.d_k) * self.normKH(
            self.W_C1k(C_cat)).view(batch_size, -1, self.num_heads, self.d_k)
        v = self.normVE(self.W_v(state)).view(batch_size, -1, self.num_heads, self.d_k) * self.normVH(
            self.W_C1v(C_cat)).view(batch_size, -1, self.num_heads, self.d_k)

        # Reshape hidden state for context
        # print('H',H.shape)
        # H = H.view(-1, self.patch_length, self.dff)
        batch_size = state.shape[0]


        # Reshape for multi-head attention
        q, k, v = [x.transpose(1, 2) for x in (q, k, v)]  # [batch_size, num_heads, seq_len, d_k]

        # Compute attention and get values
        attn_values, attn_weights = self.calculate_attention(q, k, v)
        attn_values = attn_values.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)

        # Residual connection and dropout
        Z1 = state + self.dropout1(attn_values)

        # Feed-forward network with residual connection
        Z2_norm = self.norm1(Z1)
        Z3 = F.gelu(self.linear1(Z2_norm))
        Z4 = self.linear2(Z3)
        dZ = self.dropout1(attn_values) + self.dropout2(Z4)
        Z5 = state + dZ

        return Z5

    def calculate_attention(self, q, k, v):
        """
        Calculate scaled dot-product attention.

        Args:
            q: Query tensor
            k: Key tensor
            v: Value tensor

        Returns:
            attention_output: Output after applying attention to values
            attention_weights: The attention weights
        """
        # Compute attention scores with scaling
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))

        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)

        # Compute attention output
        attention_output = torch.matmul(attention_weights, v)

        return attention_output, attention_weights