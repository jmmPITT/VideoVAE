import os
import cv2
import numpy as np
import requests
import patoolib
from tqdm import tqdm
import argparse

def download_file(url, target_path):
    """Downloads a file from a URL to a target path, showing progress."""
    if os.path.exists(target_path):
        print(f"{target_path} already exists. Skipping download.")
        return
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an exception for bad status codes
    total_size = int(response.headers.get('content-length', 0))
    with open(target_path, 'wb') as f, tqdm(
        desc=target_path,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            size = f.write(chunk)
            bar.update(size)

def extract_archive(archive_path, extract_dir):
    """Extracts a rar archive to a specified directory."""
    if os.path.exists(extract_dir):
        print(f"Extraction directory {extract_dir} already exists. Skipping extraction.")
        return
    print(f"Extracting {archive_path} to {extract_dir}...")
    patoolib.extract_archive(archive_path, outdir=extract_dir)
    print("Extraction complete.")

def main(args):
    # 1. Download the dataset
    print("Step 1: Downloading UCF101 dataset...")
    download_file(args.dataset_url, args.rar_path)

    # 2. Extract the dataset
    print("\nStep 2: Extracting dataset...")
    extract_archive(args.rar_path, args.extract_path)

    # 3. Process videos into numpy array
    print("\nStep 3: Processing videos...")
    video_folder_path = os.path.join(args.extract_path, "UCF-101")
    video_files = []
    for root, dirs, files in os.walk(video_folder_path):
        for file in files:
            if file.endswith(".avi"):
                video_files.append(os.path.join(root, file))

    all_sequences = []
    resolution = (args.width, args.height)
    for video_path in tqdm(video_files, desc="Processing all videos"):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            resized_frame = cv2.resize(frame, resolution)
            frames.append(resized_frame)
        cap.release()

        # Chunk frames into sequences of num_frames
        if len(frames) >= args.num_frames:
            for i in range(0, len(frames) - args.num_frames + 1, args.num_frames):
                sequence = np.array(frames[i:i + args.num_frames])
                all_sequences.append(sequence)

    processed_data = np.array(all_sequences)

    # 4. Save the processed data
    print(f"\nStep 4: Saving processed data to {args.output_npy_file}...")
    np.save(args.output_npy_file, processed_data)
    print(f"Data saved with shape: {processed_data.shape}")
    print("\nData preparation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and process the UCF101 dataset.")
    parser.add_argument('--dataset_url', type=str, default="https://www.crcv.ucf.edu/data/UCF101/UCF101.rar",
                        help='URL of the UCF101 dataset.')
    parser.add_argument('--rar_path', type=str, default="UCF101.rar",
                        help='Path to save the downloaded .rar file.')
    parser.add_argument('--extract_path', type=str, default="UCF101",
                        help='Directory to extract the dataset to.')
    parser.add_argument('--output_npy_file', type=str, default="ucf101_50_frames.npy",
                        help='Path to save the processed .npy file.')
    parser.add_argument('--num_frames', type=int, default=50,
                        help='Number of frames per video sequence.')
    parser.add_argument('--width', type=int, default=320,
                        help='Width of the resized frames.')
    parser.add_argument('--height', type=int, default=240,
                        help='Height of the resized frames.')
    args = parser.parse_args()
    main(args)