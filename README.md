# Video Variational Autoencoder (VAE) for UCF101

This repository contains a PyTorch implementation of a complex, transformer-based Variational Autoencoder (VAE) for video processing, specifically designed for the UCF101 dataset. The model leverages a hierarchical structure with custom LSTM cells and attention mechanisms to learn representations of video sequences.

## Features

- **Hierarchical Transformer-Based VAE**: A deep and complex model for learning video representations.
- **Configurable Data Preparation**: A flexible script to download and process the UCF101 dataset.
- **Configurable Training Script**: Easily configure hyperparameters, file paths, and model loading from the command line.
- **Perceptual Loss**: Integrates VGG-based perceptual loss to improve the visual quality of reconstructed videos.
- **Memory Optimization**: Uses gradient checkpointing to allow for training large models with larger batch sizes.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

## Usage

The process involves two main steps: preparing the dataset and training the model.

### 1. Prepare the UCF101 Dataset

The `prepare_ucf101.py` script handles the downloading, extraction, and processing of the UCF101 dataset into a `.npy` file suitable for training.

**Basic Usage:**
```bash
python prepare_ucf101.py
```

This will download the dataset, extract it, and create a `ucf101_50_frames.npy` file in the root directory.

**Command-Line Arguments:**
You can customize the data preparation process using the following arguments:

- `--dataset_url`: URL of the UCF101 dataset.
- `--rar_path`: Path to save the downloaded `.rar` file.
- `--extract_path`: Directory to extract the dataset to.
- `--output_npy_file`: Path to save the processed `.npy` file.
- `--num_frames`: Number of frames per video sequence.
- `--width`: Width of the resized frames.
- `--height`: Height of the resized frames.

**Example:**
```bash
python prepare_ucf101.py --output_npy_file data/processed_ucf101.npy --num_frames 60
```

### 2. Train the Model

The `main.py` script trains the Video VAE model.

**Basic Usage:**
You must provide the path to the `.npy` file created in the previous step.

```bash
python main.py --npy_file ucf101_50_frames.npy
```

**Command-Line Arguments:**

- `--npy_file` (required): Path to the processed `.npy` dataset file.
- `--model_path`: Path to save or load the model. (Default: `transformer_model.pth`)
- `--batch_size`: Batch size for training. (Default: 1)
- `--learning_rate`: Learning rate for the optimizer. (Default: 1e-5)
- `--num_epochs`: Number of training epochs. (Default: 50)
- `--load_model`: A flag to load a pre-trained model from the specified `model_path`.

**Example (with custom parameters):**
```bash
python main.py \
    --npy_file data/processed_ucf101.npy \
    --model_path models/my_video_vae.pth \
    --batch_size 4 \
    --learning_rate 1e-4 \
    --num_epochs 100
```

**Example (loading a pre-trained model):**
```bash
python main.py \
    --npy_file data/processed_ucf101.npy \
    --model_path models/my_video_vae_epoch_50.pth \
    --load_model
```

## Model Architecture

The core of this repository is the `TransformerNetwork` in `video_vae/model.py`. It is a sophisticated, multi-layered network composed of three main parts:

1.  **Encoder**: Processes input video frames through a convolutional encoder and a series of hierarchical layers with custom LSTM cells and attention mechanisms to produce a latent representation.
2.  **Decoder**: Reconstructs video frames from the latent representation, using a similar hierarchical structure.
3.  **Generator**: Generates new video frames from a learned distribution, also using a multi-layered refinement process.

The model uses extensive gradient checkpointing to manage memory consumption during training, making it possible to train such a deep architecture.