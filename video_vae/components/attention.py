"""
This file contains the attention mechanisms for the model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModernTransformerBlock(nn.Module):
    """
    A modern transformer block that uses scaled dot-product attention.

    This block processes an input state with a self-attention mechanism,
    optionally incorporating context tensors. It is designed to be efficient
    by using `F.scaled_dot_product_attention`.
    """

    def __init__(self, input_dims=64, patch_length=12 * 16 + 1, dff=256,
                 dropout=0.1, num_heads=2, num_context_tensors=0):
        super().__init__()
        self.input_dims = input_dims
        self.patch_length = patch_length
        self.d_model = self.input_dims
        self.num_heads = num_heads
        self.dff = dff
        self.dropout = dropout
        self.num_context_tensors = num_context_tensors

        self.d_k = self.d_model // self.num_heads

        self.W_q = nn.Linear(self.d_model, self.d_k * self.num_heads)
        self.W_k = nn.Linear(self.d_model, self.d_k * self.num_heads)
        self.W_v = nn.Linear(self.d_model, self.d_k * self.num_heads)

        if self.num_context_tensors > 0:
            self.W_Cq = nn.Linear(self.dff * self.num_context_tensors, self.d_k * self.num_heads)
            self.W_Ck = nn.Linear(self.dff * self.num_context_tensors, self.d_k * self.num_heads)
            self.W_Cv = nn.Linear(self.dff * self.num_context_tensors, self.d_k * self.num_heads)
            self.normCH = nn.LayerNorm(self.d_model)

        self.normQE = nn.LayerNorm(self.d_model)
        self.normKE = nn.LayerNorm(self.d_model)
        self.normVE = nn.LayerNorm(self.d_model)

        self.linear1 = nn.Linear(self.num_heads * self.d_k, self.d_model * 4)
        self.linear2 = nn.Linear(self.d_model * 4, self.d_model)
        self.dropout1 = nn.Dropout(self.dropout)

        self.norm1 = nn.LayerNorm(self.d_model)
        self.dropout2 = nn.Dropout(self.dropout)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, *context_tensors):
        """
        Forward pass of the transformer block.

        Args:
            state (torch.Tensor): The input state tensor.
            *context_tensors (torch.Tensor): Optional context tensors.

        Returns:
            torch.Tensor: The processed state tensor.
            torch.Tensor: The delta tensor.
            torch.Tensor: The original state tensor.
        """
        state = state.view(-1, self.patch_length, self.input_dims)
        batch_size = state.shape[0]

        q = self.normQE(self.W_q(state)).view(batch_size, -1, self.num_heads, self.d_k)
        k = self.normKE(self.W_k(state)).view(batch_size, -1, self.num_heads, self.d_k)
        v = self.normVE(self.W_v(state)).view(batch_size, -1, self.num_heads, self.d_k)

        if self.num_context_tensors > 0:
            if len(context_tensors) != self.num_context_tensors:
                raise ValueError(f"Expected {self.num_context_tensors} context tensors, but got {len(context_tensors)}")

            reshaped_contexts = []
            for ct in context_tensors:
                reshaped_contexts.append(ct.view(-1, self.patch_length, self.dff))

            C = torch.cat(reshaped_contexts, dim=-1)

            q = q * self.normCH(self.W_Cq(C)).view(batch_size, -1, self.num_heads, self.d_k)
            k = k * self.normCH(self.W_Ck(C)).view(batch_size, -1, self.num_heads, self.d_k)
            v = v * self.normCH(self.W_Cv(C)).view(batch_size, -1, self.num_heads, self.d_k)

        q, k, v = [x.transpose(1, 2) for x in (q, k, v)]

        attn_values = F.scaled_dot_product_attention(
            query=q, key=k, value=v, attn_mask=None, dropout_p=0.0, is_causal=False
        )
        attn_values = attn_values.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)

        Z1 = state + self.dropout1(attn_values)

        Z2_norm = self.norm1(Z1)
        Z3 = F.gelu(self.linear1(Z2_norm), approximate='tanh')
        Z4 = self.linear2(Z3)
        dZ = self.dropout1(attn_values) + self.dropout2(Z4)
        Z5 = state + dZ

        return Z5, dZ, state


class LegacyTransformerBlock(nn.Module):
    """
    A legacy transformer block that processes an input state with a
    self-attention mechanism, incorporating context from hidden states.

    This block is based on the original implementation and uses a manual
    attention calculation.
    """

    def __init__(self, input_dims=64, patch_length=12 * 16 + 1, dff=256,
                 dropout=0.1, num_heads=2, num_context_tensors=4):
        super().__init__()
        self.input_dims = input_dims
        self.patch_length = patch_length
        self.d_model = self.input_dims
        self.num_heads = num_heads
        self.dff = dff
        self.dropout = dropout
        self.num_context_tensors = num_context_tensors

        self.d_k = self.d_model // self.num_heads

        self.W_q = nn.Linear(self.d_model, self.d_k * self.num_heads)
        self.W_k = nn.Linear(self.d_model, self.d_k * self.num_heads)
        self.W_v = nn.Linear(self.d_model, self.d_k * self.num_heads)

        self.W_Cq = nn.Linear(self.dff * self.num_context_tensors, self.d_k * self.num_heads)
        self.W_Ck = nn.Linear(self.dff * self.num_context_tensors, self.d_k * self.num_heads)
        self.W_Cv = nn.Linear(self.dff * self.num_context_tensors, self.d_k * self.num_heads)

        self.normQE = nn.LayerNorm(self.d_model)
        self.normKE = nn.LayerNorm(self.d_model)
        self.normVE = nn.LayerNorm(self.d_model)
        self.normQH = nn.LayerNorm(self.d_model)
        self.normKH = nn.LayerNorm(self.d_model)
        self.normVH = nn.LayerNorm(self.d_model)

        self.linear1 = nn.Linear(self.num_heads * self.d_k, self.d_model * 4)
        self.linear2 = nn.Linear(self.d_model * 4, self.d_model)
        self.dropout1 = nn.Dropout(self.dropout)

        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)
        self.dropout2 = nn.Dropout(self.dropout)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, *context_tensors):
        """
        Forward pass of the legacy transformer block.

        Args:
            state (torch.Tensor): The input state tensor.
            *context_tensors (torch.Tensor): The context tensors.

        Returns:
            torch.Tensor: The processed state tensor.
        """
        state = state.view(-1, self.patch_length, self.input_dims)

        reshaped_contexts = []
        for ct in context_tensors:
            reshaped_contexts.append(ct.view(-1, self.patch_length, self.dff))

        C_cat = torch.cat(reshaped_contexts, dim=-1)
        batch_size = state.shape[0]

        q = self.normQE(self.W_q(state)).view(batch_size, -1, self.num_heads, self.d_k) * \
            self.normQH(self.W_Cq(C_cat)).view(batch_size, -1, self.num_heads, self.d_k)
        k = self.normKE(self.W_k(state)).view(batch_size, -1, self.num_heads, self.d_k) * \
            self.normKH(self.W_Ck(C_cat)).view(batch_size, -1, self.num_heads, self.d_k)
        v = self.normVE(self.W_v(state)).view(batch_size, -1, self.num_heads, self.d_k) * \
            self.normVH(self.W_Cv(C_cat)).view(batch_size, -1, self.num_heads, self.d_k)

        q, k, v = [x.transpose(1, 2) for x in (q, k, v)]

        attn_values, _ = self.calculate_attention(q, k, v)
        attn_values = attn_values.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)

        Z1 = state + self.dropout1(attn_values)

        Z2_norm = self.norm1(Z1)
        Z3 = F.gelu(self.linear1(Z2_norm))
        Z4 = self.linear2(Z3)
        dZ = self.dropout1(attn_values) + self.dropout2(Z4)
        Z5 = state + dZ

        return Z5

    def calculate_attention(self, q, k, v):
        """
        Calculates scaled dot-product attention.

        Args:
            q (torch.Tensor): Query tensor.
            k (torch.Tensor): Key tensor.
            v (torch.Tensor): Value tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the
                attention output and attention weights.
        """
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, v)
        return attention_output, attention_weights