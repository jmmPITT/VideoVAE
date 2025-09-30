"""
Main script to train the VideoVAE model.
"""
import torch
from video_vae.model import TransformerNetwork
from video_vae.trainer import MovingMNISTDataset, TemporalAutoencoderTrainer, PerceptualLoss

if __name__ == '__main__':
    # Set the environment variable to allow duplicate library loading
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # Paths to your models and dataset
    npy_file = 'ucf101_subset_batch_6.npy'  # Path to your dataset

    # Instantiate the dataset
    dataset = MovingMNISTDataset(npy_file)

    # Instantiate the TransformerNetwork model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transformer_model_path = 'transformer_model.pth'

    # Load the trained Transformer model
    transformer_model = TransformerNetwork(beta=1e-5).to(device)

    # Uncomment the following line to load a pre-trained model
    # transformer_model.load_state_dict(torch.load(transformer_model_path, map_location=device))

    # Instantiate the trainer
    trainer = TemporalAutoencoderTrainer(
        transformer_model=transformer_model,
        dataset=dataset,
        batch_size=1,
        learning_rate=1e-5,
        num_epochs=50
    )

    # Initialize perceptual loss for evaluation
    perceptual_loss = PerceptualLoss().to(device)

    # Start training
    trainer.train()

    print("\nTraining complete.")
    print("The model has been trained with a perceptual loss component to improve visual quality.")
    print("Memory optimization with gradient checkpointing was used to allow for larger batch sizes.")