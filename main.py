"""
Main script to train the VideoVAE model.
"""
import torch
import argparse
import os
from video_vae.model import TransformerNetwork
from video_vae.trainer import VideoDataset, TemporalAutoencoderTrainer

def main(args):
    # Set the environment variable to allow duplicate library loading.
    # This is a workaround for a known issue with some environments where multiple
    # OpenMP libraries can be loaded, causing a conflict.
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # Instantiate the dataset
    dataset = VideoDataset(args.npy_file)

    # Instantiate the TransformerNetwork model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transformer_model = TransformerNetwork(beta=args.learning_rate).to(device)

    if args.load_model:
        print(f"Loading pre-trained model from {args.model_path}")
        transformer_model.load_state_dict(torch.load(args.model_path, map_location=device))

    # Instantiate the trainer
    trainer = TemporalAutoencoderTrainer(
        transformer_model=transformer_model,
        dataset=dataset,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        model_save_path=args.model_path
    )

    # Start training
    trainer.train()

    print("\nTraining complete.")
    print("The model has been trained with a perceptual loss component to improve visual quality.")
    print("Memory optimization with gradient checkpointing was used to allow for larger batch sizes.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the VideoVAE model.")
    parser.add_argument('--npy_file', type=str, required=True,
                        help='Path to the processed .npy dataset file.')
    parser.add_argument('--model_path', type=str, default='transformer_model.pth',
                        help='Path to save or load the model.')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                        help='Learning rate for the optimizer.')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of training epochs.')
    parser.add_argument('--load_model', action='store_true',
                        help='Load a pre-trained model from the specified model_path.')

    args = parser.parse_args()
    main(args)