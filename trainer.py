# Filename: temporal_autoencoder_trainer.py

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

# Import the trained VAE model
from networkV5 import *  # Ensure this is the file containing your TransformerNetwork class


# Add the perceptual loss class for evaluation
class PerceptualLoss(nn.Module):
    def __init__(self, resize=True, normalize=True):
        super(PerceptualLoss, self).__init__()
        # Use weights parameter instead of deprecated pretrained parameter
        weights = models.VGG16_Weights.IMAGENET1K_V1
        self.vgg = models.vgg16(weights=weights).features[:29].eval()  # Use features up to conv5_3
        # Freeze VGG parameters to avoid training them
        for param in self.vgg.parameters():
            param.requires_grad = False

        self.resize = resize
        self.normalize = normalize
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)) if resize else nn.Identity(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) if normalize else nn.Identity()
        ])

        # MSE Loss for feature comparison
        self.criterion = nn.MSELoss()

    def forward(self, x, y):
        """
        Calculate perceptual loss between x and y using VGG features
        x, y: tensors of shape [batch_size, channels, height, width]
        """
        if x.shape[1] == 1:  # Handle grayscale input
            x = x.repeat(1, 3, 1, 1)
            y = y.repeat(1, 3, 1, 1)

        # Resize and normalize inputs if needed
        if self.resize or self.normalize:
            x = self.transform(x)
            y = self.transform(y)

        # Get VGG features
        x_features = self.vgg(x)
        y_features = self.vgg(y)

        # Calculate MSE loss between features
        return self.criterion(x_features, y_features)


import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def generate_positional_encoding(seq_len, latent_dim):
    position = torch.arange(seq_len).unsqueeze(1)  # Shape: (seq_len, 1)
    div_term = torch.exp(torch.arange(0, latent_dim, 2) * -(math.log(10000.0) / latent_dim))
    pe = torch.zeros(seq_len, latent_dim)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # Shape: (seq_len, latent_dim)


def kl_divergence(ZMU, ZLOGVAR, mu, log_var):
    """
    Compute the KL divergence between two Gaussian distributions.

    Args:
        ZMU: Predicted mean (batch_size, num_patches, seq_length, latent_dim)
        ZLOGVAR: Predicted log variance (batch_size, num_patches, seq_length, latent_dim)
        mu: True mean from VAE (batch_size, num_patches, seq_length, latent_dim)
        log_var: True log variance from VAE (batch_size, num_patches, seq_length, latent_dim)

    Returns:
        KL loss (scalar)
    """
    # Convert log variances to variances
    ZVAR = torch.exp(ZLOGVAR)  # Predicted variance
    true_var = torch.exp(log_var)  # True variance

    # Compute KL divergence term-wise
    kl_loss = 0.5 * (
            torch.log(true_var / ZVAR) - 1 + ZVAR / true_var + (ZMU - mu) ** 2 / true_var
    )

    # Reduce across all dimensions (batch_size, num_patches, seq_length, latent_dim)
    return kl_loss.mean()  # Or sum if you want total KL divergence


def add_noise(latent_vectors, timesteps, beta_schedule):
    # latent_vectors: [batch_size, latent_dim]
    # timesteps: [batch_size] integer time steps
    # beta_schedule: predefined noise schedule

    alphas = 1.0 - beta_schedule
    alphas_cumprod = np.cumprod(alphas)

    alpha_t = alphas_cumprod[timesteps]
    noise = torch.randn_like(latent_vectors)
    noisy_latent = torch.sqrt(alpha_t) * latent_vectors + torch.sqrt(1 - alpha_t) * noise
    return noisy_latent, noise


class TimeEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super(TimeEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.fc = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, t):
        # t: [batch_size]
        half_dim = self.embedding_dim // 2
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -np.log(10000) / (half_dim - 1))
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=1)
        return self.fc(emb)


# Custom Dataset for Moving MNIST
class MovingMNISTDataset(Dataset):
    def __init__(self, npy_file):
        # 1) mmap so we never load all sequences into RAM
        self.data = np.load(npy_file, mmap_mode='r')
        print('dataset shape', self.data.shape)  # e.g. (817, 25, 240, 320, 3)

        # 2) replicate your original: only use the first 500 sequences
        # self.data = self.data[:800]

        # now unpack dims
        self.num_sequences, self.num_frames, H, W, self.channels = self.data.shape

        # patch specs
        self.patch_h, self.patch_w = 15, 20
        self.grid_h, self.grid_w = H // self.patch_h, W // self.patch_w
        self.num_patches = self.grid_h * self.grid_w  # =256

    def __len__(self):
        # exactly your original patched_data.shape[0]
        return self.num_sequences

    def __getitem__(self, idx):
        # pull just that one sequence and normalize
        video = (self.data[idx].astype(np.float32) / 255.0)
        # video shape: (25, 240, 320, 3)

        # vectorized patching:
        # step 1: (T, grid_h, patch_h, grid_w, patch_w, C)
        v = video.reshape(
            self.num_frames,
            self.grid_h, self.patch_h,
            self.grid_w, self.patch_w,
            self.channels
        )
        # step 2: swap axes to (T, grid_h, grid_w, patch_h, patch_w, C)
        v = v.transpose(0, 1, 3, 2, 4, 5)
        # step 3: flatten grid_h × grid_w → num_patches: (T, 256, 15, 20, 3)
        patches = v.reshape(
            self.num_frames,
            self.num_patches,
            self.patch_h,
            self.patch_w,
            self.channels
        )

        return torch.from_numpy(patches)



# Trainer Class for Temporal Autoencoder
class TemporalAutoencoderTrainer:
    def __init__(self, transformer_model, dataset, batch_size=16, learning_rate=1e-3,
                 num_epochs=10):
        self.transformer_model = transformer_model
        self.dataset = dataset
        sample = self.dataset[0]  # Get the first sample
        # print('Sample shape:', sample.shape)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Move transformer model to device
        self.transformer_model.to(self.device)
        self.optimizer_TAE = optim.Adam(self.transformer_model.parameters(), lr=self.learning_rate, betas=(0.9, 0.99),
                                eps=1e-8, weight_decay=1e-4)

        # DataLoader
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def train(self):
        # with torch.no_grad():

        for epoch in range(1, self.num_epochs + 1):
            epoch_loss = 0.0
            for batch_idx, sequences in enumerate(self.dataloader):
                sequences = sequences.to(self.device)  # Shape: (batch_size, num_patches, num_frames, 1, 64, 64)
                # print('sequences',sequences.shape)
                batch_size, num_frames, num_patch, height, width, channels = sequences.size()

                # Initialize lists to store latent vectors for each sequence in the batch
                latent_sequences = []
                model_inputs = []

                # Encode each frame using the VAE encoder to get latent vectors
                with torch.no_grad():
                    for i in range(num_frames):
                        # print('Sequences',sequences[:, i, :, :, :].shape)
                        frames = sequences[:, i, :, :, :].view(-1, 256,
                                                               15 * 20 * 3)  # Shape: (batch_size, 1, 64, 64)

                        latent_sequences.append(frames.unsqueeze(1))  # Shape: (batch_size, 1, latent_dim)

                # Stack latent vectors to form sequences
                latent_sequences = torch.cat(latent_sequences, dim=1)  # Shape: (batch_size, num_frames, latent_dim)
                # print('latent_sequences',latent_sequences.shape)
                # Now, we need to process each sequence in the batch
                # For simplicity, we'll process one sequence at a time (can be optimized for batch processing)

                total_loss = 0.0
                # for b in range(batch_size):
                # Split latent sequence into patches if needed
                # Assuming that you have already split your images into patches during VAE training
                # If not, you need to adjust the latent_sequence accordingly

                # Initialize recurrent states (C, M, H, N) to zeros
                # Assuming you have three layers in your LSTM as per your model definition
                C_F1 = torch.zeros(batch_size, self.transformer_model.num_patch_one, self.transformer_model.dff1,
                                   requires_grad=True).to(
                    self.device)
                C_F2 = torch.zeros(batch_size, self.transformer_model.num_patch_two, self.transformer_model.dff2,
                                   requires_grad=True).to(
                    self.device)
                C_F3 = torch.zeros(batch_size, self.transformer_model.num_patch_three, self.transformer_model.dff3,
                                   requires_grad=True).to(
                    self.device)
                C_F4 = torch.zeros(batch_size, self.transformer_model.num_patch_four, self.transformer_model.dff4,
                                   requires_grad=True).to(
                    self.device)

                # Prepare the input state (latent_sequence)
                # Reshape latent_sequence to match input dimensions expected by the TransformerNetwork
                # For example, reshape to (1, patch_length, input_dims)
                # Assuming patch_length = num_patches (e.g., 4), input_dims = latent_dim

                # Here, we need to handle the splitting into patches if necessary
                # For simplicity, let's assume we have one patch (patch_length = 1)

                # Adjust as needed based on your actual implementation

                # For demonstration, let's proceed with patch_length = 1
                ### This is the input and the target ###
                state = latent_sequences.view(batch_size, 50, 256,
                                              15 * 20 * 3)  # Shape: (1, num_frames, latent_dim)

                ### collect a running loss through the timesteps ###
                total_loss = 0
                recon_loss = 0
                Z_latents_list = []
                Z_mu_list = []
                Z_var_list = []
                C1_Forward_List = []
                C2_Forward_List = []
                C3_Forward_List = []
                C4_Forward_List = []
                heatmap_list = np.zeros((50, 10, 16, 16))
                # Loop over 25 timesteps (you could vectorize this if desired)
                for t in range(50):
                    # Extract the state at timestep t and reshape appropriately.
                    input_state = state[:, t, :].view(-1, 256, 15 * 20 * 3)

                    # Pass through transformer model's encoder.
                    # Note: Ensure your encoder returns Z, Zmu, and Zvar.
                    Z, C_F1, C_F2, C_F3, C_F4= self.transformer_model.encoder(
                        input_state, C_F1, C_F2, C_F3, C_F4, t)


                    
                    # Append latent samples, means, and variances (or log variances) for later use.
                    C1_Forward_List.append(C_F1.view(-1, 1, self.transformer_model.num_patch_one, self.transformer_model.dff1))
                    C2_Forward_List.append(C_F2.view(-1, 1, self.transformer_model.num_patch_two, self.transformer_model.dff2))
                    C3_Forward_List.append(C_F3.view(-1, 1, self.transformer_model.num_patch_three, self.transformer_model.dff3))
                    C4_Forward_List.append(C_F4.view(-1, 1, self.transformer_model.num_patch_four, self.transformer_model.dff4))



                    # Once we have processed all timesteps...
                    if t == 49:

                        
                        # Concatenate along the time dimension.
                        # Pass the concatenated latent samples to the decoder.
                        out, C1_backwards, C2_backwards, C3_backwards, C4_backwards, mu1, log_var1, mu2, log_var2, mu3, log_var3, mu4, log_var4 = self.transformer_model.decoder(Z, C_F1, C_F2, C_F3, C_F4, t+5)

                        ### Compute VAE Loss ###
                        # Compute the KL Divergence loss term
                        # This measures the difference between the encoder's distribution and a standard normal distribution
                        kl_loss = -0.5 * torch.mean(1 + log_var1 - mu1.pow(2) - log_var1.exp())
                        kl_loss += -0.5 * torch.mean(1 + log_var2 - mu2.pow(2) - log_var2.exp())
                        kl_loss += -0.5 * torch.mean(1 + log_var3 - mu3.pow(2) - log_var3.exp())
                        kl_loss += -0.5 * torch.mean(1 + log_var4 - mu4.pow(2) - log_var4.exp())
    

                        ### Now output losses ###
                        z_target_ = state.view(batch_size, num_frames, 256, 15 * 20 * 3)
                        z_target_ = z_target_.view(batch_size, num_frames, 16, 16, 15, 20, 3)    # your existing reshaping
                        z_target_ = z_target_.permute(0,1,2, 4,3,5,6).contiguous()
                        z_target_ =z_target_.view(batch_size, 50, 240, 320, 3)
                        out = out.view(-1, 55, 240, 320, 3)
                        
                        recon_loss += F.mse_loss(out[:,0:50], z_target_[:,0:50])
                        total_loss += recon_loss*10.0 + kl_loss*0.01

                        self.optimizer_TAE.zero_grad()
                        total_loss.backward()
                        self.optimizer_TAE.step()
                        
                if batch_idx % 10 == 0:
                    print(
                        f'Epoch [{epoch}/{self.num_epochs}] Batch [{batch_idx}/{len(self.dataloader)}] '
                        f'TAELoss: {total_loss.item():.6f}', f'recon_loss: {recon_loss.item():.6f}', f'kl_loss: {kl_loss.item():.6f}')
                if batch_idx % 100 == 0:
                    self.save_model_TAE(epoch)
                            


    def split_into_patches(self, frame):
        patches = []
        for i in range(8):
            for j in range(8):
                patch = frame[8 * i:8 * (i + 1), 8 * j:8 * (j + 1)]
                patch = patch[np.newaxis, :, :]  # Add channel dimension
                patches.append(patch)
        patches = np.stack(patches, axis=0)  # Shape: (4, 1, 32, 32)
        return patches

    def reconstruct_from_patches(self, patches):
        # patches shape: (4, 1, 32, 32)
        # print('patches shape',patches.shape)
        reconstructed_frame = np.zeros((1, 240, 320, 3), dtype=np.float32)
        c = 0
        for i in range(16):
            for j in range(16):
                reconstructed_frame[:, 15 * i:15 * (i + 1), 20 * j:20 * (j + 1), :] = patches[c].view(15, 20,
                                                                                                      3).cpu().numpy()
                c += 1
        return reconstructed_frame  # Shape: (1, 64, 64)

    def display_sequence(self, original_seq, t=1):
        """
        Plots two rows:
          - Top row: original_seq
          - Bottom row: reconstructed_seq
        """
        num_frames = original_seq.shape[1]
        print(original_seq.shape)
        # Let’s create the figure & subplots ourselves
        fig, axes = plt.subplots(
            nrows=1,
            ncols=num_frames,
            figsize=(num_frames * 1.2, 6)  # Make it wide enough to fit 50 frames
        )

        # This reduces the default padding between subplots
        # so images are closer together
        plt.subplots_adjust(
            wspace=0.01,  # horizontal space
            hspace=0.05  # vertical space
        )

        for j in range(1):

            for i in range(num_frames):
                # Plot original on top row
                ax_top = axes[i]
                print(original_seq.shape)
                ax_top.imshow(original_seq[0, i])
                ax_top.axis('off')
                if i == num_frames // 2:
                    ax_top.set_title('Original Sequence', fontsize=12)

                # Optionally draw the green rectangle for frame t
                if i == t:
                    height, width = original_seq[0, i].shape[:2]
                    rect = patches.Rectangle(
                        (0, 0), width, height,
                        linewidth=2, edgecolor='green', facecolor='none'
                    )
                    ax_top.add_patch(rect)

                # Plot reconstructed on bottom row
                # ax_bottom = axes[1, i]
                # ax_bottom.imshow(reconstructed_seq[i])
                # ax_bottom.axis('off')
                # if i == num_frames // 2:
                #     ax_bottom.set_title('Reconstructed Sequence', fontsize=12)

        # Save & show
        plt.tight_layout()  # Might slightly tweak spacing, but we already used subplots_adjust
        plt.savefig('sequence_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

    def save_model_TAE(self, epoch):
        # Save the model state dict
        model_save_path = f'transformer_model.pth'
        torch.save(self.transformer_model.state_dict(), model_save_path)
        print(f'Model saved at {model_save_path}')

    def save_model_GAN(self, epoch):
        # Save the model state dict
        model_save_path = f'gan_model.pth'
        torch.save(self.gan_model.state_dict(), model_save_path)
        print(f'Model saved at {model_save_path}')

    def load_model(self, epoch):
        # Load the model state dict
        model_save_path = f'transformer_model_epoch_{epoch}.pth'
        self.transformer_model.load_state_dict(torch.load(model_save_path, map_location=self.device))
        self.transformer_model.to(self.device)
        self.transformer_model.eval()
        print(f'Model loaded from {model_save_path}')


# Main function to run the training
if __name__ == '__main__':
    # Paths to your models and dataset
    npy_file = 'ucf101_subset_batch_6.npy'  # Path to your dataset

    # Instantiate the dataset
    dataset = MovingMNISTDataset(npy_file)

    # Instantiate the TransformerNetwork model
    # transformer_model = TransformerNetwork(beta=1e-4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transformer_model_path = 'transformer_model.pth'
    # transformer_model.load_state_dict(torch.load(transformer_model_path, map_location=device))
    # Load the trained Transformer model (Diffusion model)
    transformer_model = TransformerNetwork(beta=1e-5).to(device)
    # transformer_model.load_state_dict(torch.load(transformer_model_path, map_location=device))

    # gan_model.load_state_dict(torch.load(gan_model_path, map_location=device))
    # Load the complete state dictionary
    # checkpoint = torch.load(transformer_model_path, map_location=device)

    # Remove the specific keys that are causing the error

    # Load the filtered state dictionary into the model
    # transformer_model.load_state_dict(checkpoint)
    # transformer_model.eval()  # Set Transformer to evaluation mode
    # Instantiate the trainer
    trainer = TemporalAutoencoderTrainer(
        transformer_model=transformer_model,
        dataset=dataset,
        batch_size=1,
        learning_rate=1e-5,
        num_epochs=50  # Adjust as needed
    )

    # Initialize perceptual loss for evaluation
    perceptual_loss = PerceptualLoss().to(device)

    # Start training (which also tests the model)
    trainer.train()

    # Print a message to explain what we've added
    print("\nPerceptual loss has been added to the training process.")
    print("This should improve the visual quality of reconstructions by focusing on")
    print("higher-level features rather than just pixel-wise differences.")
    print("The weight of the perceptual loss component can be adjusted using the perceptual_weight parameter.")

    print("\nMemory optimization with gradient checkpointing has been implemented.")
    print("This trades computation for memory by not storing all intermediate activations")
    print("and recomputing them during the backward pass, allowing training with larger batch sizes.")
    print("You can monitor GPU memory usage during training to see the benefits.")
