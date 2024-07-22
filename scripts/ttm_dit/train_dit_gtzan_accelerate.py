import argparse
from pathlib import Path
import librosa
import numpy as np
import torch
from torch import nn
from accelerate import Accelerator
import torch.optim as optim
import wandb
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from einops import rearrange

wandb.require("core")

from audidata.io import load
from audidata.tokenizers import BaseTokenizer
from audidata.datasets import GTZAN
from audidata.samplers import InfiniteSampler
from audidata.transforms import RandomCrop
from mugen.lm.gpt2 import GPTConfig, GPT
from mugen.lm.llama import LLaMAConfig, LLaMA
from mugen.diffusion import create_diffusion

from codec import CodecWrapper

from mugen.models.dit import DiT


def train(args):

    # Arguments
    model_name = args.model_name

    # Default parameters
    sr = 44100
    clip_duration = 10.
    batch_size = 8
    num_workers = 16
    pin_memory = True
    learning_rate = 1e-4
    test_step_frequency = 1000
    save_step_frequency = 5000
    training_steps = 200000
    wandb_log = True
    device = "cuda"
    evaluate_num = 10

    codec_dim = 1024  # Codec vocab
    n_quantizers = 2

    filename = Path(__file__).stem

    checkpoints_dir = Path("./checkpoints", filename, model_name)

    root = "/datasets/gtzan"

    if wandb_log:
        wandb.init(project="music_generation_dit") 

    # Crop the first a few seconds
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

    test_dataset = GTZAN(
        root=root,
        split="train",
        test_fold=0,
        sr=sr,
        transform=audio_transform
    )

    # Sampler
    train_sampler = InfiniteSampler(train_dataset)
    
    # Dataloader
    train_dataloader = DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        sampler=train_sampler,
        num_workers=num_workers, 
        pin_memory=pin_memory
    )

    # Codec
    codec_model = CodecWrapper.load_model()

    # Language model
    model = get_model(model_name=model_name)

    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # Prepare for multiprocessing
    accelerator = Accelerator()
    
    codec_model, model, diffusion, optimizer, train_dataloader = accelerator.prepare(
        codec_model, model, diffusion, optimizer, train_dataloader)
    
    # Create checkpoints directory
    Path(checkpoints_dir).mkdir(parents=True, exist_ok=True)

    # Train
    for step, data in enumerate(tqdm(train_dataloader)):

        # labels = data["label"]
        # shape: (batch_size,)
        target = data["target"]

        audio = data["audio"]
        # shape: (batch_size, audio_samples)
 
        # Encode audio to discrete codes
        codes, latents = CodecWrapper.encode(
            audio=audio, 
            model=codec_model.module, 
            n_quantizers=n_quantizers
        )
        # shape: (batch_size, codes_num)

        t = torch.randint(0, diffusion.num_timesteps, (audio.shape[0],), device=device)

        target = torch.argmax(target, dim=-1)
        model_kwargs = {"y": target}
        model.train()
        loss_dict = diffusion.training_losses(model, latents, t, model_kwargs)
        loss = loss_dict["loss"].mean()

        optimizer.zero_grad()   # Reset all parameter.grad to 0
        accelerator.backward(loss)  # Update all parameter.grad
        optimizer.step()  # Update all parameters based on all parameter.grad

        if step % test_step_frequency == 0:
            print(step, loss.item())

        # Save model
        if step % save_step_frequency == 0:

            accelerator.wait_for_everyone()

            if accelerator.is_main_process:

                unwrapped_model = accelerator.unwrap_model(model)

                checkpoint_path = Path(checkpoints_dir, "step={}.pth".format(step))
                torch.save(unwrapped_model.state_dict(), checkpoint_path)
                print("Save model to {}".format(checkpoint_path))

                checkpoint_path = Path(checkpoints_dir, "latest.pth")
                torch.save(unwrapped_model.state_dict(), Path(checkpoint_path))
                print("Save model to {}".format(checkpoint_path))

        if step == training_steps:
            break


def get_model(model_name: str):

    if model_name == "dit":

        model = DiT(
            input_size=861,
            patch_size=1,
            in_channels=16,
            hidden_size=384,
            depth=28,
            num_heads=16,
            mlp_ratio=4.0,
            class_dropout_prob=0.1,
            num_classes=10,
            learn_sigma=True,
        )

    else:
        raise NotImplementedError

    return model


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="dit")
    args = parser.parse_args()

    train(args)