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



class TransformerNetwork(nn.Module):
    def __init__(self, beta, input_dims=32, hidden_dim=128, fc1_dims=64, fc2_dims=32, n_actions=4,
                 name='transformer', chkpt_dir='td3_MAT'):
        super(TransformerNetwork, self).__init__()
        self.input_dims = (15 * 20 * 3)
        self.hidden_dim = hidden_dim
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.patch_length = 256
        # self.seq_length = 31
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = 'tmp/transformer_model_Long10H.pth'
        self.checkpoint_file = name + '_td3'
        self.d_model = self.input_dims
        self.num_heads = 1
        self.dff = 512
        self.dropout = 0.1
        dropout = self.dropout

        self.dffEncoder = 1024
        self.dff1 = 128
        self.dff2 = 256
        self.dff3 = 512
        self.dff4 = 1024

        self.num_patch_one = 1024
        self.num_patch_two = 256
        self.num_patch_three = 64
        self.num_patch_four = 16
        ### 1024*32 + 256 * 64 + 64 * 128 + 16 * 512

        self.image_encoder = ConvEncoder16x16(n_c=self.dffEncoder)

        ### Encoder ###
        self.enc_embedding = nn.Parameter(torch.zeros(1, self.num_patch_one, self.dffEncoder))

        # Transformer encoder (assumed to be a nn.Module)
        self.TFE1 = TransformerBlock4(
            input_dims=self.dffEncoder,
            patch_length=self.num_patch_one,
            dff=self.dff1,
            dropout=dropout,
            num_heads=16
        )

        self.lstmE1 = CustomLSTMCell3(
            patch_size=self.num_patch_one,
            d_model=self.dffEncoder,
            dff=self.dff1
        )



        self.conv_ZE12 = nn.Conv2d(
            in_channels=self.dffEncoder,
            out_channels=self.dff2,
            kernel_size=3,
            stride=2,
            padding=1,  # to keep output = floor((16+2*1−3)/2+1) = 8
            bias=False
        )
        self.norm_ZE12 = nn.LayerNorm(self.dff2)  # or LayerNorm, your preference
        self.act_ZE12 = nn.ReLU(inplace=True)

        # self.norm_ZE23 = nn.LayerNorm(4*self.dff2)
        # self.patch_reduce_ZE23 = nn.Linear(4*self.dff2, self.dff3)
        self.conv_ZE23 = nn.Conv2d(
            in_channels=self.dff2,
            out_channels=self.dff3,
            kernel_size=3,
            stride=2,
            padding=1,  # to keep output = floor((16+2*1−3)/2+1) = 8
            bias=False
        )
        self.norm_ZE23 = nn.LayerNorm(self.dff3)  # or LayerNorm, your preference
        self.act_ZE23 = nn.ReLU(inplace=True)

        # self.norm_ZE34 = nn.LayerNorm(4*self.dff3)
        # self.patch_reduce_ZE34 = nn.Linear(4*self.dff3, self.dff4)

        self.conv_ZE34 = nn.Conv2d(
            in_channels=self.dff3,
            out_channels=self.dff4,
            kernel_size=3,
            stride=2,
            padding=1,  # to keep output = floor((16+2*1−3)/2+1) = 8
            bias=False
        )
        self.norm_ZE34 = nn.LayerNorm(self.dff4)  # or LayerNorm, your preference
        self.act_ZE34 = nn.ReLU(inplace=True)

        # self.norm_E12 = nn.LayerNorm(4*self.dff1)
        # self.patch_reduce_E12 = nn.Linear(4*self.dff1, self.dff2)
        self.conv_E12 = nn.Conv2d(
            in_channels=self.dff1,
            out_channels=self.dff2,
            kernel_size=3,
            stride=2,
            padding=1,  # to keep output = floor((16+2*1−3)/2+1) = 8
            bias=False
        )
        self.norm_E12 = nn.LayerNorm(self.dff2)  # or LayerNorm, your preference
        self.act_E12 = nn.ReLU(inplace=True)

    
        self.conv_E21 = nn.ConvTranspose2d(
            in_channels=self.dff2,
            out_channels=self.dff1,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False
        )
        self.norm_E21 = nn.LayerNorm(self.dff1)  # or LayerNorm, your preference
        self.act_E21 = nn.ReLU(inplace=True)

        # self.TE21 = nn.TransformerEncoderLayer(
        #     d_model=128,
        #     nhead=8,
        #     dim_feedforward=1024,
        #     dropout=0.01,
        #     activation='gelu'  # Activation function ('relu' or 'gelu')
        # )
        # self.Epos21_embedding = nn.Parameter(torch.zeros(1, self.num_patch_one, 128))

        self.lstmE2 = CustomLSTMCell2(
            patch_size=self.num_patch_two,
            d_model=self.dff2,
            dff=self.dff2
        )

        # self.norm_E23 = nn.LayerNorm(4*self.dff2)
        # self.patch_reduce_E23 = nn.Linear(4*self.dff2, self.dff3)
        self.conv_E23 = nn.Conv2d(
            in_channels=self.dff2,
            out_channels=self.dff3,
            kernel_size=3,
            stride=2,
            padding=1,  # to keep output = floor((16+2*1−3)/2+1) = 8
            bias=False
        )
        self.norm_E23 = nn.LayerNorm(self.dff3)  # or LayerNorm, your preference
        self.act_E23 = nn.ReLU(inplace=True)

        # self.norm_E32 = nn.LayerNorm(256)
        # self.patch_expand_E32 = nn.Linear(256, self.dff2)
        self.conv_E32 = nn.ConvTranspose2d(
            in_channels=self.dff3,
            out_channels=self.dff2,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False
        )
        self.norm_E32 = nn.LayerNorm(self.dff2)  # or LayerNorm, your preference
        self.act_E32 = nn.ReLU(inplace=True)
        # self.TE32 = nn.TransformerEncoderLayer(
        #     d_model=256,
        #     nhead=16,
        #     dim_feedforward=1024,
        #     dropout=0.01,
        #     activation='gelu'  # Activation function ('relu' or 'gelu')
        # )
        # self.Epos32_embedding = nn.Parameter(torch.zeros(1, self.num_patch_two, 256))

        # self.norm_E31 = nn.LayerNorm(64)
        # self.patch_expand_E31 = nn.Linear(64, self.dff1)

        self.conv_E31 = nn.ConvTranspose2d(
            in_channels=self.dff2,
            out_channels=self.dff1,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False
        )
        self.norm_E31 = nn.LayerNorm(self.dff1)  # or LayerNorm, your preference
        self.act_E31 = nn.ReLU(inplace=True)

        self.lstmE3 = CustomLSTMCell1(
            patch_size=self.num_patch_three,
            d_model=self.dff3,
            dff=self.dff3
        )

        # self.norm_E34 = nn.LayerNorm(4*self.dff3)
        # self.patch_reduce_E34 = nn.Linear(4*self.dff3, self.dff4)
        self.conv_E34 = nn.Conv2d(
            in_channels=self.dff3,
            out_channels=self.dff4,
            kernel_size=3,
            stride=2,
            padding=1,  # to keep output = floor((16+2*1−3)/2+1) = 8
            bias=False
        )
        self.norm_E34 = nn.LayerNorm(self.dff4)  # or LayerNorm, your preference
        self.act_E34 = nn.ReLU(inplace=True)

        # self.norm_E43 = nn.LayerNorm(1024)
        # self.patch_expand_E43 = nn.Linear(1024, self.dff3)

        self.conv_E43 = nn.ConvTranspose2d(
            in_channels=self.dff4,
            out_channels=self.dff3,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False
        )
        self.norm_E43 = nn.LayerNorm(self.dff3)  # or LayerNorm, your preference
        self.act_E43 = nn.ReLU(inplace=True)

        # self.norm_E42 = nn.LayerNorm(256)
        # self.patch_expand_E42 = nn.Linear(256, self.dff2)
        self.conv_E42 = nn.ConvTranspose2d(
            in_channels=self.dff3,
            out_channels=self.dff2,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False
        )
        self.norm_E42 = nn.LayerNorm(self.dff2)  # or LayerNorm, your preference
        self.act_E42 = nn.ReLU(inplace=True)

        # self.norm_E41 = nn.LayerNorm(64)
        # self.patch_expand_E41 = nn.Linear(64, self.dff1)
        self.conv_E41 = nn.ConvTranspose2d(
            in_channels=self.dff2,
            out_channels=self.dff1,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False
        )
        self.norm_E41 = nn.LayerNorm(self.dff1)  # or LayerNorm, your preference
        self.act_E41 = nn.ReLU(inplace=True)

        # self.TE41 = nn.TransformerEncoderLayer(
        #     d_model=64,
        #     nhead=8,
        #     dim_feedforward=1024,
        #     dropout=0.01,
        #     activation='gelu'  # Activation function ('relu' or 'gelu')
        # )
        # self.Epos41_embedding = nn.Parameter(torch.zeros(1, self.num_patch_one, 64))

        self.lstmE4 = CustomLSTMCell0(
            patch_size=self.num_patch_four,
            d_model=self.dff4,
            dff=self.dff4
        )

        ### Decoder ###
        self.dec_embedding = nn.Parameter(torch.zeros(1, self.num_patch_one, self.input_dims))
        #
        # # Encoder
        # self.TE4MU = nn.TransformerEncoderLayer(
        #     d_model=self.dff4,
        #     nhead=32,
        #     dim_feedforward=2048,
        #     dropout=0.01,
        #     activation='gelu'  # Activation function ('relu' or 'gelu')
        # )
        #
        # self.TE4VAR = nn.TransformerEncoderLayer(
        #     d_model=self.dff4,
        #     nhead=32,
        #     dim_feedforward=2048,
        #     dropout=0.01,
        #     activation='gelu'  # Activation function ('relu' or 'gelu')
        # )
        #
        # self.TE3MU = nn.TransformerEncoderLayer(
        #     d_model=self.dff3,
        #     nhead=16,
        #     dim_feedforward=2048,
        #     dropout=0.01,
        #     activation='gelu'  # Activation function ('relu' or 'gelu')
        # )
        #
        # self.TE3VAR = nn.TransformerEncoderLayer(
        #     d_model=self.dff3,
        #     nhead=16,
        #     dim_feedforward=2048,
        #     dropout=0.01,
        #     activation='gelu'  # Activation function ('relu' or 'gelu')
        # )
        #
        # self.TE2MU = nn.TransformerEncoderLayer(
        #     d_model=self.dff2,
        #     nhead=8,
        #     dim_feedforward=2048,
        #     dropout=0.01,
        #     activation='gelu'  # Activation function ('relu' or 'gelu')
        # )
        #
        # self.TE2VAR = nn.TransformerEncoderLayer(
        #     d_model=self.dff2,
        #     nhead=8,
        #     dim_feedforward=2048,
        #     dropout=0.01,
        #     activation='gelu'  # Activation function ('relu' or 'gelu')
        # )
        #
        # self.TE1MU = nn.TransformerEncoderLayer(
        #     d_model=self.dff1,
        #     nhead=4,
        #     dim_feedforward=2048,
        #     dropout=0.01,
        #     activation='gelu'  # Activation function ('relu' or 'gelu')
        # )
        #
        # self.TE1VAR = nn.TransformerEncoderLayer(
        #     d_model=self.dff1,
        #     nhead=4,
        #     dim_feedforward=2048,
        #     dropout=0.01,
        #     activation='gelu'  # Activation function ('relu' or 'gelu')
        # )
        self.fc_mu4 = nn.Linear(self.dff4, self.dff4)

        self.fc_log_var4 = nn.Linear(self.dff4, self.dff4)

        self.fc_mu3 = nn.Linear(self.dff3, self.dff3)
        #
        self.fc_log_var3 = nn.Linear(self.dff3, self.dff3)
        #
        self.fc_mu2 = nn.Linear(self.dff2, self.dff2)
        #
        self.fc_log_var2 = nn.Linear(self.dff2, self.dff2)
        #
        self.fc_mu1 = nn.Linear(self.dff1, self.dff1)
        #
        self.fc_log_var1 = nn.Linear(self.dff1, self.dff1)


        # Transformer encoder (assumed to be a nn.Module)
        self.TFD1 = TransformerBlock6(
            input_dims=self.input_dims,
            patch_length=self.num_patch_one,
            dff=self.dff1,
            dropout=dropout,
            num_heads=10
        )

        self.lstmD1 = CustomLSTMCell5(
            patch_size=self.num_patch_one,
            d_model=self.input_dims,
            dff=self.dff1
        )

        # self.norm_D12 = nn.LayerNorm(4*self.dff1)
        # self.patch_reduce_D12 = nn.Linear(4*self.dff1, self.dff2)
        self.conv_D12 = nn.Conv2d(
            in_channels=self.dff1,
            out_channels=self.dff2,
            kernel_size=3,
            stride=2,
            padding=1,  # to keep output = floor((16+2*1−3)/2+1) = 8
            bias=False
        )
        self.norm_D12 = nn.LayerNorm(self.dff2)  # or LayerNorm, your preference
        self.act_D12 = nn.ReLU(inplace=True)

        # self.norm_D21 = nn.LayerNorm(128)
        # self.patch_expand_D21 = nn.Linear( 128, self.dff1)
        self.conv_D21 = nn.ConvTranspose2d(
            in_channels=self.dff2,
            out_channels=self.dff1,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False
        )
        self.norm_D21 = nn.LayerNorm(self.dff1)  # or LayerNorm, your preference
        self.act_D21 = nn.ReLU(inplace=True)

        self.lstmD2 = CustomLSTMCell4(
            patch_size=self.num_patch_two,
            d_model=self.dff2,
            dff=self.dff2
        )

        # self.norm_D23 = nn.LayerNorm(4*self.dff2)
        # self.patch_reduce_D23 = nn.Linear(4*self.dff, self.dff3)
        self.conv_D23 = nn.Conv2d(
            in_channels=self.dff2,
            out_channels=self.dff3,
            kernel_size=3,
            stride=2,
            padding=1,  # to keep output = floor((16+2*1−3)/2+1) = 8
            bias=False
        )
        self.norm_D23 = nn.LayerNorm(self.dff3)  # or LayerNorm, your preference
        self.act_D23 = nn.ReLU(inplace=True)

        # self.norm_D32 = nn.LayerNorm(256)
        # self.patch_expand_D32 = nn.Linear(256, self.dff2)
        self.conv_D32 = nn.ConvTranspose2d(
            in_channels=self.dff3,
            out_channels=self.dff2,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False
        )
        self.norm_D32 = nn.LayerNorm(self.dff2)  # or LayerNorm, your preference
        self.act_D32 = nn.ReLU(inplace=True)

        # self.norm_D31 = nn.LayerNorm(64)
        # self.patch_expand_D31 = nn.Linear(64, self.dff1)
        self.conv_D31 = nn.ConvTranspose2d(
            in_channels=self.dff2,
            out_channels=self.dff1,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False
        )
        self.norm_D31 = nn.LayerNorm(self.dff1)  # or LayerNorm, your preference
        self.act_D31 = nn.ReLU(inplace=True)

        self.lstmD3 = CustomLSTMCell3(
            patch_size=self.num_patch_three,
            d_model=self.dff3,
            dff=self.dff3
        )

        # self.norm_D34 = nn.LayerNorm(4*self.dff3)
        # self.patch_reduce_D34 = nn.Linear(4*self.dff3, self.dff4)
        self.conv_D34 = nn.Conv2d(
            in_channels=self.dff3,
            out_channels=self.dff4,
            kernel_size=3,
            stride=2,
            padding=1,  # to keep output = floor((16+2*1−3)/2+1) = 8
            bias=False
        )
        self.norm_D34 = nn.LayerNorm(self.dff4)  # or LayerNorm, your preference
        self.act_D34 = nn.ReLU(inplace=True)

        # self.norm_D43 = nn.LayerNorm(1024)
        # self.patch_expand_D43 = nn.Linear(1024, self.dff3)
        self.conv_D43 = nn.ConvTranspose2d(
            in_channels=self.dff4,
            out_channels=self.dff3,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False
        )
        self.norm_D43 = nn.LayerNorm(self.dff3)  # or LayerNorm, your preference
        self.act_D43 = nn.ReLU(inplace=True)

        # self.norm_D42 = nn.LayerNorm(256)
        # self.patch_expand_D42 = nn.Linear(256, self.dff2)
        self.conv_D42 = nn.ConvTranspose2d(
            in_channels=self.dff3,
            out_channels=self.dff2,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False
        )
        self.norm_D42 = nn.LayerNorm(self.dff2)  # or LayerNorm, your preference
        self.act_D42 = nn.ReLU(inplace=True)

        # self.norm_D41 = nn.LayerNorm(64)
        # self.patch_expand_D41 = nn.Linear(64, self.dff1)
        self.conv_D41 = nn.ConvTranspose2d(
            in_channels=self.dff2,
            out_channels=self.dff1,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False
        )

        self.norm_D41 = nn.LayerNorm(self.dff1)  # or LayerNorm, your preference
        self.act_D41 = nn.ReLU(inplace=True)

        self.lstmD4 = CustomLSTMCell2(
            patch_size=self.num_patch_four,
            d_model=self.dff4,
            dff=self.dff4
        )

        ### Generator ###
        self.gen_embedding = nn.Parameter(torch.zeros(1, self.num_patch_one, self.input_dims))

        # Transformer encoder (assumed to be a nn.Module)
        self.TFG1 = TransformerBlock5(
            input_dims=self.input_dims,
            patch_length=self.num_patch_one,
            dff=self.dff1,
            dropout=dropout,
            num_heads=10
        )

        self.conv_G12 = nn.Conv2d(
            in_channels=self.dff1,
            out_channels=self.dff2,
            kernel_size=3,
            stride=2,
            padding=1,  # to keep output = floor((16+2*1−3)/2+1) = 8
            bias=False
        )
        self.norm_G12 = nn.LayerNorm(self.dff2)  # or LayerNorm, your preference
        self.act_G12 = nn.ReLU(inplace=True)



        self.conv_G21 = nn.ConvTranspose2d(
            in_channels=self.dff2,
            out_channels=self.dff1,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False
        )
        self.norm_G21 = nn.LayerNorm(self.dff1)  # or LayerNorm, your preference
        self.act_G21 = nn.ReLU(inplace=True)

        self.conv_G23 = nn.Conv2d(
            in_channels=self.dff2,
            out_channels=self.dff3,
            kernel_size=3,
            stride=2,
            padding=1,  # to keep output = floor((16+2*1−3)/2+1) = 8
            bias=False
        )
        self.norm_G23 = nn.LayerNorm(self.dff3)  # or LayerNorm, your preference
        self.act_G23 = nn.ReLU(inplace=True)

        self.conv_G32 = nn.ConvTranspose2d(
            in_channels=self.dff3,
            out_channels=self.dff2,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False
        )
        self.norm_G32 = nn.LayerNorm(self.dff2)  # or LayerNorm, your preference
        self.act_G32 = nn.ReLU(inplace=True)

        self.conv_G31 = nn.ConvTranspose2d(
            in_channels=self.dff2,
            out_channels=self.dff1,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False
        )
        self.norm_G31 = nn.LayerNorm(self.dff1)  # or LayerNorm, your preference
        self.act_G31 = nn.ReLU(inplace=True)

        self.conv_G34 = nn.Conv2d(
            in_channels=self.dff3,
            out_channels=self.dff4,
            kernel_size=3,
            stride=2,
            padding=1,  # to keep output = floor((16+2*1−3)/2+1) = 8
            bias=False
        )
        self.norm_G34 = nn.LayerNorm(self.dff4)  # or LayerNorm, your preference
        self.act_G34 = nn.ReLU(inplace=True)

        self.conv_G43 = nn.ConvTranspose2d(
            in_channels=self.dff4,
            out_channels=self.dff3,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False
        )
        self.norm_G43 = nn.LayerNorm(self.dff3)  # or LayerNorm, your preference
        self.act_G43 = nn.ReLU(inplace=True)

        self.conv_G42 = nn.ConvTranspose2d(
            in_channels=self.dff3,
            out_channels=self.dff2,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False
        )
        self.norm_G42 = nn.LayerNorm(self.dff2)  # or LayerNorm, your preference
        self.act_G42 = nn.ReLU(inplace=True)

        # self.norm_G41 = nn.LayerNorm(64)
        # self.patch_expand_G41 = nn.Linear(64, self.dff1)

        self.conv_G41 = nn.ConvTranspose2d(
            in_channels=self.dff2,
            out_channels=self.dff1,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False
        )
        self.norm_G41 = nn.LayerNorm(self.dff1)  # or LayerNorm, your preference
        self.act_G41 = nn.ReLU(inplace=True)

        self.lstmG1 = CustomLSTMCell4(
            patch_size=self.num_patch_one,
            d_model=self.input_dims,
            dff=self.dff1
        )

        self.lstmG2 = CustomLSTMCell3(
            patch_size=self.num_patch_two,
            d_model=self.dff2,
            dff=self.dff2
        )

        self.lstmG3 = CustomLSTMCell2(
            patch_size=self.num_patch_three,
            d_model=self.dff3,
            dff=self.dff3
        )

        self.lstmG4 = CustomLSTMCell1(
            patch_size=self.num_patch_four,
            d_model=self.dff4,
            dff=self.dff4
        )



        # your feature‐extractor convs (no change)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(4, 32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, 64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=2, dilation=2)
        self.norm3 = nn.GroupNorm(16, 128)
        self.conv4 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.norm4 = nn.GroupNorm(8, 64)
        self.conv5 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.norm5 = nn.GroupNorm(4, 32)

        # — Squeeze‑and‑Excite on the 128‑dim features from conv3 —
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # (B,128,1,1)
        self.se_fc1 = nn.Conv2d(128, 128 // 16, 1)  # → (B,8,1,1)
        self.se_fc2 = nn.Conv2d(128 // 16, 128, 1)  # → (B,128,1,1)

        self.act_out = nn.ReLU(inplace=True)

        # FINAL conv: keeps your 3‑channel output but now **stride=2** halves H and W
        self.conv6 = nn.Conv2d(32, 3, kernel_size=3, padding=1, stride=2)

        self.optimizer = optim.Adam(self.parameters(), lr=beta, betas=(0.9, 0.99), eps=1e-8, weight_decay=1e-4)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def _patch_expand(self, C, batch, HpatchIN, WpatchIN, HpatchOUT, WpatchOUT, conv, norm, act):
        """
        Upsample from nin → nout patches.
        x: [B, nin, D]
        returns: [B, nout, D]
        """

        # break
        # print('C in', C.shape)

        Cb = C.view(batch, HpatchIN, WpatchIN, -1)
        B, H, W, D = Cb.shape

        # → [B, D, H, W]
        x = Cb.permute(0, 3, 1, 2).contiguous()
        x = conv(x)
        # x = norm(x)
        # x = act(x)

        # Cb = Cb.view(batch, hblks*wblks, bh*bw*D)
        # apply your reduce layer → [B, hblks*wblks, D_out]
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(batch, HpatchOUT * WpatchOUT, -1)
        x = norm(x)
        x = act(x)
        # print('x out', x.shape)

        # back to [B,nout,D]
        return x.view(batch, HpatchOUT * WpatchOUT, -1)

    def _patch_reduce(self, C, batch, HpatchIN, WpatchIN, HpatchOUT, WpatchOUT, conv, norm, act):
        """
        Block-reduce from (hblks×bh)×(wblks×bw) patches down to hblks*wblks = nout.
        C: [B, hblks*bh*wblks*bw, D]
        returns: [B, nout, D_out]
        """
        D = C.size(-1)
        # carve into blocks
        # print('C size', C.shape)

        # break
        Cb = C.view(batch, HpatchIN, WpatchIN, -1)
        B, H, W, D = Cb.shape

        # → [B, D, H, W]
        x = Cb.permute(0, 3, 1, 2).contiguous()
        x = conv(x)
        # x = norm(x)
        # x = act(x)

        # Cb = Cb.view(batch, hblks*wblks, bh*bw*D)
        # apply your reduce layer → [B, hblks*wblks, D_out]
        x = x.permute(0, 2, 3, 1).contiguous()
        # print('x out', x.shape)
        x = x.view(batch, HpatchOUT * WpatchOUT, -1)
        x = norm(x)
        x = act(x)

        return x.view(batch, HpatchOUT * WpatchOUT, -1)

    def _encoder_step1(self, state, C1, C2, C3, C4):
        """First step of encoder with checkpointing"""
        # During inference, don't use checkpointing
        batch_size = state.shape[0]

        state = state.view(batch_size, 256, 15 * 20 * 3)
        state = state.view(batch_size, 16, 16, 15, 20, 3)  # your existing reshaping
        state = state.permute(0, 1, 3, 2, 4, 5).contiguous()
        state = state.view(batch_size, 240, 320, 3)
        state = state.permute(0, 3, 1, 2).contiguous()

        # state = self.image_encoder(state)
        state = checkpoint.checkpoint(self.image_encoder, state, use_reentrant=False)
        # print('state',state.shape)
        state = state.view(-1, self.dffEncoder, self.num_patch_one)
        state = state.permute(0, 2, 1).contiguous()

        state = state.view(-1, self.num_patch_one, self.dffEncoder) + self.enc_embedding

        # print("C2 shape fucker", C2.shape)
        C2UP = self._patch_expand(C2, batch_size,
                                  16, 16,  # e.g. 4,3
                                  32, 32,  # e.g. 4,4
                                  self.conv_E21,
                                  self.norm_E21, self.act_E21)
        # print("HI")
        C32UP = self._patch_expand(C3, batch_size,
                                   8, 8,  # e.g. 4,3
                                   16, 16,  # e.g. 4,4
                                   self.conv_E32,
                                   self.norm_E32, self.act_E32)

        # print('C32UP.shape', C32UP.shape)
        # C32UP_ = self.norm_E32(C32UP)

        C31UP = self._patch_expand(C32UP, batch_size,
                                   16, 16,  # e.g. 4,3
                                   32, 32,  # e.g. 4,4
                                   self.conv_E31,
                                   self.norm_E31, self.act_E31)

        C43UP = self._patch_expand(C4, batch_size,
                                   4, 4,  # e.g. 4,3
                                   8, 8,  # e.g. 4,4
                                   self.conv_E43,
                                   self.norm_E43, self.act_E43)

        C42UP = self._patch_expand(C43UP, batch_size,
                                   8, 8,  # e.g. 4,3
                                   16, 16,  # e.g. 4,4
                                   self.conv_E42,
                                   self.norm_E42, self.act_E42)

        C41UP = self._patch_expand(C42UP, batch_size,
                                   16, 16,  # e.g. 4,3
                                   32, 32,  # e.g. 4,4
                                   self.conv_E41,
                                   self.norm_E41, self.act_E41)

        # C3 = C3 + C3UP

        Z = self.TFE1(state, C1, C2UP, C31UP, C41UP)

        ### Create Z-Down Skips ###
        Z2 = self._patch_reduce(state, batch_size,
                                32, 32,  # e.g. 4,3
                                16, 16,  # e.g. 4,4
                                self.conv_ZE12,
                                self.norm_ZE12, self.act_ZE12)

        Z3 = self._patch_reduce(Z2, batch_size,
                                16, 16,  # e.g. 4,3
                                8, 8,  # e.g. 4,4
                                self.conv_ZE23,
                                self.norm_ZE23, self.act_ZE23)

        Z4 = self._patch_reduce(Z3, batch_size,
                                8, 8,  # e.g. 4,3
                                4, 4,  # e.g. 4,4
                                self.conv_ZE34,
                                self.norm_ZE34, self.act_ZE34)

        C1 = self.lstmE1(Z, C1, C2UP, C31UP, C41UP)

        # now reduce C1→C2
        C1_blocks = self._patch_reduce(C1, batch_size,
                                       32, 32,  # e.g. 4,3
                                       16, 16,  # e.g. 4,4
                                       self.conv_E12,
                                       self.norm_E12, self.act_E12)
        # print("I made it here")
        C2 = self.lstmE2(C1_blocks + Z2, C2, C32UP, C42UP)

        # then reduce C2→C3
        C2_blocks = self._patch_reduce(C2, batch_size,
                                       16, 16,  # hblks=2,bh=2
                                       8, 8,  # wblks=2,bw=2
                                       self.conv_E23,
                                       self.norm_E23, self.act_E23)

        C3 = self.lstmE3(C2_blocks + Z3, C3, C43UP)

        # then reduce C2→C3
        C3_blocks = self._patch_reduce(C3, batch_size,
                                       8, 8,  # hblks=2,bh=2
                                       4, 4,  # wblks=2,bw=2
                                       self.conv_E34,
                                       self.norm_E34, self.act_E34)

        # print('C3_blocks', C3_blocks.shape)

        C4 = self.lstmE4(C3_blocks + Z4, C4)

        return Z, C1, C2, C3, C4

    def encoder(self, state, C1, C2, C3, C4, t):
        # Use checkpointing for each transformer block to save memory
        batch_size = state.shape[0]

        # if self.training:
        # First transformer block with checkpointing
        Z, C1, C2, C3, C4 = checkpoint.checkpoint(
            self._encoder_step1,
            state, C1, C2, C3, C4,
            preserve_rng_state=True,
            use_reentrant=False
        )

        # Z, C_sample_, logCvar_, Cvar_, Cmu_ = self.generator_step(Z, C1, C2, C3)

        return Z, C1, C2, C3, C4

    def _decoder_step(self, state, CE1, CE2, CE3, CE4, C1, C2, C3, C4, CG1, CG2, CG3, CG4, batch_size, t):
        """Single decoder step with checkpointing"""
        # During inference, no checkpointing
        # selected_embedding = self.dec_embedding  # Shape: [patch_length, embedding_dim]

        C2UP = self._patch_expand(C2, batch_size,
                                  16, 16,  # e.g. 4,3
                                  32, 32,  # e.g. 4,4
                                  self.conv_D21,
                                  self.norm_D21, self.act_D21)

        C32UP = self._patch_expand(C3, batch_size,
                                   8, 8,  # e.g. 4,3
                                   16, 16,  # e.g. 4,4
                                   self.conv_D32,
                                   self.norm_D32, self.act_D32)
        # C32UP_ = self.norm_D32(C32UP)

        # print('C32UP.shape', C32UP.shape)

        C31UP = self._patch_expand(C32UP, batch_size,
                                   16, 16,  # e.g. 4,3
                                   32, 32,  # e.g. 4,4
                                   self.conv_D31,
                                   self.norm_D31, self.act_D31)

        C43UP = self._patch_expand(C4, batch_size,
                                   4, 4,  # e.g. 4,3
                                   8, 8,  # e.g. 4,4
                                   self.conv_D43,
                                   self.norm_D43, self.act_D43)

        C42UP = self._patch_expand(C43UP, batch_size,
                                   8, 8,  # e.g. 4,3
                                   16, 16,  # e.g. 4,4
                                   self.conv_D42,
                                   self.norm_D42, self.act_D42)

        C41UP = self._patch_expand(C42UP, batch_size,
                                   16, 16,  # e.g. 4,3
                                   32, 32,  # e.g. 4,4
                                   self.conv_D41,
                                   self.norm_D41, self.act_D41)

        # C3 = C3 + C3UP
        # print("state", state.shape)

        Z = self.TFD1(state, CE1, C1, CG1, C2UP, C31UP, C41UP)
        C1 = self.lstmD1(Z, C1, CE1, CG1, C2UP, C31UP, C41UP)

        # now reduce C1→C2
        C1_blocks = self._patch_reduce(C1, batch_size,
                                       32, 32,  # e.g. 4,3
                                       16, 16,  # e.g. 4,4
                                       self.conv_D12,
                                       self.norm_D12, self.act_D12)

        C2 = self.lstmD2(C1_blocks, C2, CE2, CG2, C32UP, C42UP)

        # then reduce C2→C3
        C2_blocks = self._patch_reduce(C2, batch_size,
                                       16, 16,  # hblks=2,bh=2
                                       8, 8,  # wblks=2,bw=2
                                       self.conv_D23,
                                       self.norm_D23, self.act_D23)

        C3 = self.lstmD3(C2_blocks, C3, CG3, CE3, C43UP)

        # then reduce C2→C3
        C3_blocks = self._patch_reduce(C3, batch_size,
                                       8, 8,  # hblks=2,bh=2
                                       4, 4,  # wblks=2,bw=2
                                       self.conv_D34,
                                       self.norm_D34, self.act_D34)

        C4 = self.lstmD4(C3_blocks, C4, CG4, CE4)

        return Z, C1, C2, C3, C4

    # In your TransformerNetwork class, add this new method:

    def _generator_loop_body(self, state, CD1, CD2, CD3, CD4, CG1, CG2, CG3, CG4):
        """
        This function contains the logic for a SINGLE iteration of the generator's
        refinement loop. It is designed to be called by torch.checkpoint.
        Its inputs and outputs match the state of the loop precisely.
        """
        batch_size = CG1.shape[0]

        C2UP = self._patch_expand(CG2, batch_size,
                                  16, 16,  # e.g. 4,3
                                  32, 32,  # e.g. 4,4
                                  self.conv_G21,
                                  self.norm_G21, self.act_G21)

        C32UP = self._patch_expand(CG3, batch_size,
                                   8, 8,  # e.g. 4,3
                                   16, 16,  # e.g. 4,4
                                   self.conv_G32,
                                   self.norm_G32, self.act_G32)

        C31UP = self._patch_expand(C32UP, batch_size,
                                   16, 16,  # e.g. 4,3
                                   32, 32,  # e.g. 4,4
                                   self.conv_G31,
                                   self.norm_G31, self.act_G31)

        C43UP = self._patch_expand(CG4, batch_size,
                                   4, 4,  # e.g. 4,3
                                   8, 8,  # e.g. 4,4
                                   self.conv_G43,
                                   self.norm_G43, self.act_G43)

        C42UP = self._patch_expand(C43UP, batch_size,
                                   8, 8,  # e.g. 4,3
                                   16, 16,  # e.g. 4,4
                                   self.conv_G42,
                                   self.norm_G42, self.act_G42)

        C41UP = self._patch_expand(C42UP, batch_size,
                                   16, 16,  # e.g. 4,3
                                   32, 32,  # e.g. 4,4
                                   self.conv_G41,
                                   self.norm_G41, self.act_G41)

        # C3 = C3 + C3UP
        # print("state", state.shape)

        Z = self.TFG1(state, CD1, CG1, C2UP, C31UP, C41UP)
        CG1 = self.lstmG1(Z, CD1, CG1, C2UP, C31UP, C41UP)

        # now reduce C1→C2
        C1_blocks = self._patch_reduce(CG1, batch_size,
                                       32, 32,  # e.g. 4,3
                                       16, 16,  # e.g. 4,4
                                       self.conv_G12,
                                       self.norm_G12, self.act_G12)
        # print(C1_blocks.shape, CD2.shape, CG2.shape, C32UP.shape, C42UP.shape)
        CG2 = self.lstmG2(C1_blocks, CD2, CG2, C32UP, C42UP)

        # then reduce C2→C3
        C2_blocks = self._patch_reduce(CG2, batch_size,
                                       16, 16,  # hblks=2,bh=2
                                       8, 8,  # wblks=2,bw=2
                                       self.conv_G23,
                                       self.norm_G23, self.act_G23)

        CG3 = self.lstmG3(C2_blocks, CD3, CG3, C43UP)

        # then reduce C2→C3
        C3_blocks = self._patch_reduce(CG3, batch_size,
                                       8, 8,  # hblks=2,bh=2
                                       4, 4,  # wblks=2,bw=2
                                       self.conv_G34,
                                       self.norm_G34, self.act_G34)

        CG4 = self.lstmG4(C3_blocks, CD4, CG4)


        # Z = self.TFG1(state, CD1, CD2, CD3, CD4, CG1, CG2, CG3, CG4)

        # We must return all the updated states for the next iteration,
        # PLUS the Z_refined tensor which needs to be appended to the list.
        return Z, CG1, CG2, CG3, CG4

    # In your TransformerNetwork class, REPLACE your existing generator_step with this:
    # Define this function inside your nn.Module class
    def _run_full_block(self, x):
        """Contains all the logic to be checkpointed."""
        x1 = self.act_out(self.norm1(self.conv1(x)))
        x2 = self.act_out(self.norm2(self.conv2(x1)))
        x3 = self.act_out(self.norm3(self.conv3(x2)))

        # SE block
        se = self.avgpool(x3)
        se = self.act_out(self.se_fc1(se))
        se = torch.sigmoid(self.se_fc2(se))
        x3 = x3 * se

        x4 = self.act_out(self.norm4(self.conv4(x3)))
        x5 = self.act_out(self.norm5(self.conv5(x4)))

        # 2x downsample
        out = self.conv6(x5)

        return out

    def generator_step(self, Z, CD1, CD2, CD3, CD4):
        """Final transformations in decoder with checkpointed refinement loop."""
        # --- Initial setup is identical to your code ---
        selected_embedding = self.gen_embedding
        batch_size = CD1.shape[0]

        batch_replicated_embedding = selected_embedding.expand(batch_size, -1, -1)
        Z = batch_replicated_embedding.view(-1, self.num_patch_one, self.input_dims)


        CG1 = torch.zeros(batch_size, self.num_patch_one, self.dff1, requires_grad=True).to(
            self.device)
        CG2 = torch.zeros(batch_size, self.num_patch_two, self.dff2, requires_grad=True).to(
            self.device)
        CG3 = torch.zeros(batch_size, self.num_patch_three, self.dff3, requires_grad=True).to(
            self.device)
        CG4 = torch.zeros(batch_size, self.num_patch_four, self.dff4, requires_grad=True).to(
            self.device)

        # --- The new, memory-efficient loop ---
        for _ in range(3):
            # The state variables (Z, C_G1, etc.) are passed into the checkpoint function.
            Z, CG1, CG2, CG3, CG4 = checkpoint.checkpoint(
                self._generator_loop_body,
                Z, CD1, CD2, CD3, CD4, CG1, CG2, CG3, CG4,
                preserve_rng_state=True,
                use_reentrant=False
            )


        # ——— Usage ———
        Z = Z.view(batch_size, 32, 32, 15, 20, 3)  # your existing reshaping
        Z = Z.permute(0, 1, 3, 2, 4, 5).contiguous()
        Z = Z.view(batch_size, 32 * 15, 32 * 20, 3)

        x = Z.permute(0, 3, 1, 2).contiguous()  # → (batch, 3, 240, 320)


        # Checkpoint the entire helper function
        out = checkpoint.checkpoint(self._run_full_block, x, use_reentrant=False)

        # If you need NHWC back, run permute after the checkpoint
        out = out.permute(0, 2, 3, 1).contiguous()

        # if you need NHWC back:
        # out = out.permute(0, 2, 3, 1).contiguous()

        return out, CG1, CG2, CG3, CG4

    def decoder(self, ZMUlist, CE1, CE2, CE3, CE4, t):
        batch_size = CE1.shape[0]
        outBackwards = []
        C1_backwards = []
        C2_backwards = []
        C3_backwards = []
        C4_backwards = []


        #
        mu4 = self.fc_mu4(CE4)
        log_var4 = F.softplus(self.fc_log_var4(CE4))

        mu3 = self.fc_mu3(CE3)
        log_var3 = F.softplus(self.fc_log_var3(CE3))

        mu2 = self.fc_mu2(CE2)
        log_var2 = F.softplus(self.fc_log_var2(CE2))

        mu1 = self.fc_mu1(CE1)
        log_var1 = F.softplus(self.fc_log_var1(CE1))
        #
     

        # 2. Sample from the latent distribution using the reparameterization trick
        std = torch.exp(0.5 * log_var4)          # Calculate standard deviation: sigma = exp(0.5 * log(sigma^2))
        epsilon = torch.randn_like(std)        # Sample from a standard normal distribution N(0, I)
        CE4 = mu4 + epsilon * std                 # The sampled latent vector


        
        # 2. Sample from the latent distribution using the reparameterization trick
        std = torch.exp(0.5 * log_var3)          # Calculate standard deviation: sigma = exp(0.5 * log(sigma^2))
        epsilon = torch.randn_like(std)        # Sample from a standard normal distribution N(0, I)
        CE3 = mu3 + epsilon * std                 # The sampled latent vector
        
        
        # 2. Sample from the latent distribution using the reparameterization trick
        std = torch.exp(0.5 * log_var2)          # Calculate standard deviation: sigma = exp(0.5 * log(sigma^2))
        epsilon = torch.randn_like(std)        # Sample from a standard normal distribution N(0, I)
        CE2 = mu2 + epsilon * std                 # The sampled latent vector



        # 2. Sample from the latent distribution using the reparameterization trick
        std = torch.exp(0.5 * log_var1)          # Calculate standard deviation: sigma = exp(0.5 * log(sigma^2))
        epsilon = torch.randn_like(std)        # Sample from a standard normal distribution N(0, I)
        CE1 = mu1 + epsilon * std                 # The sampled latent vector



        CG1 = torch.zeros(batch_size, self.num_patch_one, self.dff1, requires_grad=True).to(
            self.device)
        CG2 = torch.zeros(batch_size, self.num_patch_two, self.dff2, requires_grad=True).to(
            self.device)
        CG3 = torch.zeros(batch_size, self.num_patch_three, self.dff3, requires_grad=True).to(
            self.device)
        CG4 = torch.zeros(batch_size, self.num_patch_four, self.dff4, requires_grad=True).to(
            self.device)



        CD1 = torch.zeros(batch_size, self.num_patch_one, self.dff1, requires_grad=True).to(
            self.device)
        CD2 = torch.zeros(batch_size, self.num_patch_two, self.dff2, requires_grad=True).to(
            self.device)
        CD3 = torch.zeros(batch_size, self.num_patch_three, self.dff3, requires_grad=True).to(
            self.device)
        CD4 = torch.zeros(batch_size, self.num_patch_four, self.dff4, requires_grad=True).to(
            self.device)

        while t >= 0:
            # Select the t-th embedding

            selected_embedding = self.dec_embedding  # Shape: [patch_length, embedding_dim]
            batch_replicated_embedding = selected_embedding.expand(batch_size, -1, -1)
            Z = batch_replicated_embedding.view(-1, self.num_patch_one, self.input_dims)

            for _ in range(1):
                Z, CD1, CD2, CD3, CD4 = checkpoint.checkpoint(
                    self._decoder_step,
                    Z, CE1, CE2, CE3, CE4, CD1, CD2, CD3, CD4, CG1, CG2, CG3, CG4, batch_size, t,
                    preserve_rng_state=True,
                    use_reentrant=False
                )

                C1_backwards.append(CD1.view(-1, 1, self.num_patch_one, self.dff1))
                C2_backwards.append(CD2.view(-1, 1, self.num_patch_two, self.dff2))
                C3_backwards.append(CD3.view(-1, 1, self.num_patch_three, self.dff3))
                C4_backwards.append(CD4.view(-1, 1, self.num_patch_four, self.dff4))

            Z_, CG1, CG2, CG3, CG4 = self.generator_step(Z, CD1, CD2, CD3, CD4)
            outBackwards.append(Z_.view(-1, 1, 240, 320, 3))

            t = t - 1

        outBackwards = outBackwards[::-1]
        C1_backwards = C1_backwards[::-1]
        C2_backwards = C2_backwards[::-1]
        C3_backwards = C3_backwards[::-1]
        C4_backwards = C4_backwards[::-1]

        CR1_out = torch.cat(C1_backwards, dim=1)
        CR2_out = torch.cat(C2_backwards, dim=1)
        CR3_out = torch.cat(C3_backwards, dim=1)
        CR4_out = torch.cat(C4_backwards, dim=1)
        outBackwards = torch.cat(outBackwards, dim=1)

        # Reverse the list and concatenate
        out = F.sigmoid(outBackwards)

        return out, CR1_out, CR2_out, CR3_out, CR4_out, mu1, log_var1, mu2, log_var2, mu3, log_var3, mu4, log_var4

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))


