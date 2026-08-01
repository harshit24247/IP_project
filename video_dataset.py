import numpy as np
import torch
from torch.utils.data import Dataset
import scipy.io as sio
import os

class WebcamGazeDataset(Dataset):
    def __init__(self, data_dir, sequence_len=8):
        frames = np.load(f'{data_dir}/frames.npy')
        labels = np.load(f'{data_dir}/labels.npy')

        self.sequences = []
        self.labels    = []

        for i in range(len(frames) - sequence_len):
            self.sequences.append(frames[i:i+sequence_len])
            self.labels.append(labels[i+sequence_len-1])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq   = torch.tensor(self.sequences[idx], dtype=torch.float32).unsqueeze(1) / 255.0
        label = torch.tensor(self.labels[idx],    dtype=torch.float32)
        return seq, label


class MPIIGazeDataset(Dataset):
    def __init__(self, root_dir, sequence_len=8):
        self.sequences = []
        self.labels    = []
        self.seq_len   = sequence_len

        print("Loading MPIIGaze dataset...")
        for subject in sorted(os.listdir(root_dir)):
            subject_path = os.path.join(root_dir, subject)
            if not os.path.isdir(subject_path):
                continue

            for mat_file in sorted(os.listdir(subject_path)):
                if not mat_file.endswith('.mat'):
                    continue

                try:
                    data   = sio.loadmat(os.path.join(subject_path, mat_file))
                    images = data['data'][0][0]['left'][0][0]['image']
                    gazes  = data['data'][0][0]['left'][0][0]['gaze']

                    for i in range(len(images) - sequence_len):
                        self.sequences.append(images[i:i+sequence_len])
                        self.labels.append(gazes[i+sequence_len-1])
                except:
                    continue

            print(f"  ✅ {subject} loaded")

        print(f"MPIIGaze total: {len(self.sequences)} sequences")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq   = torch.tensor(self.sequences[idx], dtype=torch.float32).unsqueeze(1) / 255.0
        label = torch.tensor(self.labels[idx],    dtype=torch.float32)
        return seq, label