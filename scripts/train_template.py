import numpy as np
from torch.utils.data import Dataset, DataLoader

from audidata.datasets import GTZAN
from audidata.transforms import RandomCrop
from audidata.samplers import InfiniteSampler



def train():

    # Default parameters
    sr = 44100
    clip_duration = 10.
    batch_size = 4
    num_workers = 16
    pin_memory = True
    
    root = "/datasets/gtzan"

    # Crop the first 10 seconds
    audio_transform = RandomCrop(
        clip_duration=clip_duration,
        sr=sr
    )

    # Datasets
    train_dataset = GTZAN(
        root=root,
        split="train",
        test_fold=0,
        sr=sr,
        transform=audio_transform
    )

    # Sampler
    train_sampler = InfiniteSampler(dataset=train_dataset)
    
    # Dataloader
    train_dataloader = DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        sampler=train_sampler,
        num_workers=num_workers, 
        pin_memory=pin_memory
    )

    for step, data in enumerate(train_dataloader):

        label = data["label"]
        print(label)
        # shape: (batch_size)
        
        audio = data["audio"] 
        print(audio)
        # shape: (batch_size, audio_samples)

        # Training ...

        break


if __name__ == '__main__':

    train()