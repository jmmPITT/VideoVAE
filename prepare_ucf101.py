import os
import cv2
import numpy as np
import requests
import patoolib
from tqdm import tqdm

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

def process_videos(video_dir, num_frames=50, resolution=(320, 240)):
    """Processes videos into fixed-length, resized frame sequences."""
    all_sequences = []
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.avi')]

    for video_name in tqdm(video_files, desc="Processing videos"):
        video_path = os.path.join(video_dir, video_name)
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
        if len(frames) >= num_frames:
            for i in range(0, len(frames) - num_frames + 1, num_frames):
                sequence = np.array(frames[i:i + num_frames])
                all_sequences.append(sequence)

    return np.array(all_sequences)

def main():
    # Configuration
    dataset_url = "https://www.crcv.ucf.edu/data/UCF101/UCF101.rar"
    rar_path = "UCF101.rar"
    extract_path = "UCF101"
    video_folder_path = os.path.join(extract_path, "UCF-101")
    output_npy_file = "ucf101_50_frames.npy"

    # 1. Download the dataset
    print("Step 1: Downloading UCF101 dataset...")
    download_file(dataset_url, rar_path)

    # 2. Extract the dataset
    print("\nStep 2: Extracting dataset...")
    extract_archive(rar_path, extract_path)

    # 3. Process videos into numpy array
    print("\nStep 3: Processing videos...")
    # We need to find the actual video files. Based on the UCF101 structure, they are in subdirectories.
    # Let's find all video files in the extracted directory.
    video_files = []
    for root, dirs, files in os.walk(video_folder_path):
        for file in files:
            if file.endswith(".avi"):
                video_files.append(os.path.join(root, file))

    all_sequences = []
    for video_path in tqdm(video_files, desc="Processing all videos"):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Resize frame to 240x320
            resized_frame = cv2.resize(frame, (320, 240))
            frames.append(resized_frame)
        cap.release()

        # Chunk frames into 50-frame sequences
        if len(frames) >= 50:
            for i in range(0, len(frames) - 50 + 1, 50):
                sequence = np.array(frames[i:i + 50])
                all_sequences.append(sequence)

    processed_data = np.array(all_sequences)

    # 4. Save the processed data
    print(f"\nStep 4: Saving processed data to {output_npy_file}...")
    np.save(output_npy_file, processed_data)
    print(f"Data saved with shape: {processed_data.shape}")
    print("\nData preparation complete.")

if __name__ == "__main__":
    main()