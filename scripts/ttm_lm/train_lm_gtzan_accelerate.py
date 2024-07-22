import argparse
from pathlib import Path
import librosa
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import wandb
from accelerate import Accelerator
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

wandb.require("core")

from audidata.io import load
from audidata.tokenizers import BaseTokenizer
from audidata.datasets import GTZAN
from audidata.samplers import InfiniteSampler
from audidata.transforms import StartCrop, RandomCrop
from mugen.lm.gpt2 import GPTConfig, GPT
from mugen.lm.llama import LLaMAConfig, LLaMA

from codec import CodecWrapper
from tokenizers import get_gtzan_tokenizer
from train_lm_gtzan import get_model, codes_to_tokens, validate, forward_in_batch


def train(args):

    # Arguments
    model_name = args.model_name

    # Default parameters
    sr = 44100
    clip_duration = 10.
    batch_size = 16
    num_workers = 16
    pin_memory = True
    learning_rate = 1e-4
    test_step_frequency = 1000
    save_step_frequency = 2000
    training_steps = 100000
    wandb_log = True
    device = "cuda"
    evaluate_num = 10

    codec_dim = 1024  # Codec vocab
    n_quantizers = 1

    filename = Path(__file__).stem

    checkpoints_dir = Path("./checkpoints", filename, model_name)

    root = "/datasets/gtzan"

    if wandb_log:
        wandb.init(project="music_generation_lm") 

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
    
    # Tokenizer
    tokenizer = get_gtzan_tokenizer(
        gtzan_labels=GTZAN.labels, 
        codec_labels=range(codec_dim)
    )

    # Language model
    model = get_model(model_name=model_name, vocab_size=tokenizer.vocab_size)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # Prepare for multiprocessing
    accelerator = Accelerator()

    
    codec_model, model, optimizer, train_dataloader = accelerator.prepare(
        codec_model, model, optimizer, train_dataloader)
    
    # Create checkpoints directory
    Path(checkpoints_dir).mkdir(parents=True, exist_ok=True)

    # Train
    for step, data in enumerate(tqdm(train_dataloader)):

        labels = data["label"]
        # shape: (batch_size,)

        audio = data["audio"]
        # shape: (batch_size, audio_samples)
 
        # Encode audio to discrete codes
        codes, _ = CodecWrapper.encode(
            audio=audio, 
            model=codec_model.module, 
            n_quantizers=n_quantizers
        )
        # shape: (batch_size, codes_num)

        # Codes to tokens
        tokens = codes_to_tokens(labels=labels, codes=codes, tokenizer=tokenizer)
        # shape: (batch_size, tokens_num)

        # Prepare inputs and targets
        tokens = torch.LongTensor(tokens)
        input_tokens = tokens[:, 0 : -1]
        target_tokens = tokens[:, 1 :].contiguous()

        # Forward
        model.train()
        logits, loss = model(
            idx=input_tokens, 
            targets=target_tokens
        )

        # Optimize
        optimizer.zero_grad()   # Reset all parameter.grad to 0
        accelerator.backward(loss)     # Update all parameter.grad
        optimizer.step()    # Update all parameters based on all parameter.grad

        # Validate
        if step % test_step_frequency == 0:

            accelerator.wait_for_everyone()

            if accelerator.is_main_process:

                train_loss = validate(
                    dataset=train_dataset,
                    sr=sr,
                    clip_duration=clip_duration,
                    batch_size=batch_size,
                    codec_model=codec_model.module,
                    n_quantizers=n_quantizers,
                    tokenizer=tokenizer,
                    model=model.module,
                    evaluate_num=evaluate_num
                )

                test_loss = validate(
                    dataset=test_dataset,
                    sr=sr,
                    clip_duration=clip_duration,
                    batch_size=batch_size,
                    codec_model=codec_model.module,
                    n_quantizers=n_quantizers,
                    tokenizer=tokenizer,
                    model=model.module,
                    evaluate_num=evaluate_num
                )

                print("--- step: {} ---".format(step))
                print("Evaluate on {} songs.".format(evaluate_num))
                print("Train Loss: {:.3f}".format(train_loss))
                print("Test Loss: {:.3f}".format(test_loss))

                if wandb_log:
                    wandb.log(
                        data={
                            "train_loss": train_loss,
                            "test_loss": test_loss,
                        },
                        step=step
                    )
        
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="llama")
    args = parser.parse_args()

    train(args)