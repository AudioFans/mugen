import argparse
from pathlib import Path
import librosa
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import wandb
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

wandb.require("core")

from audidata.io import load
from audidata.tokenizers import BaseTokenizer
from audidata.datasets import GTZAN
from audidata.samplers import InfiniteSampler
from audidata.transforms import RandomCrop
from mugen.lm.gpt2 import GPTConfig, GPT
from mugen.lm.llama import LLaMAConfig, LLaMA

from codec import CodecWrapper
from tokenizers import get_gtzan_tokenizer


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
    codec_model.to(device)

    # Tokenizer
    tokenizer = get_gtzan_tokenizer(
        gtzan_labels=GTZAN.labels, 
        codec_labels=range(codec_dim)
    )

    # Language model
    model = get_model(model_name=model_name, vocab_size=tokenizer.vocab_size)
    model.to(device)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Create checkpoints directory
    Path(checkpoints_dir).mkdir(parents=True, exist_ok=True)

    # Train
    for step, data in enumerate(tqdm(train_dataloader)):

        labels = data["label"]
        # shape: (batch_size,)

        audio = data["audio"].to(device)
        # shape: (batch_size, audio_samples)
 
        # Encode audio to discrete codes
        codes, _ = CodecWrapper.encode(audio=audio, model=codec_model, n_quantizers=n_quantizers)
        # shape: (batch_size, codes_num)

        # Codes to tokens
        tokens = codes_to_tokens(labels=labels, codes=codes, tokenizer=tokenizer)
        # shape: (batch_size, tokens_num)

        # Prepare inputs and targets
        tokens = torch.LongTensor(tokens).to(device)
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
        loss.backward()     # Update all parameter.grad
        optimizer.step()    # Update all parameters based on all parameter.grad

        # Validate
        if step % test_step_frequency == 0:

            train_loss = validate(
                dataset=train_dataset,
                sr=sr,
                clip_duration=clip_duration,
                batch_size=batch_size,
                codec_model=codec_model,
                n_quantizers=n_quantizers,
                tokenizer=tokenizer,
                model=model,
                evaluate_num=evaluate_num
            )

            test_loss = validate(
                dataset=test_dataset,
                sr=sr,
                clip_duration=clip_duration,
                batch_size=batch_size,
                codec_model=codec_model,
                n_quantizers=n_quantizers,
                tokenizer=tokenizer,
                model=model,
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
            checkpoint_path = Path(checkpoints_dir, "step={}.pth".format(step))
            torch.save(model.state_dict(), checkpoint_path)
            print("Save model to {}".format(checkpoint_path))

            checkpoint_path = Path(checkpoints_dir, "latest.pth")
            torch.save(model.state_dict(), Path(checkpoint_path))
            print("Save model to {}".format(checkpoint_path))

        if step == training_steps:
            break


def get_model(model_name: str, vocab_size):

    if model_name == "gpt2":

        gptconf = GPTConfig(
            block_size=1024, 
            vocab_size=vocab_size,
            n_layer=12,
            n_head=12,
            n_embd=768
        )
        model = GPT(gptconf)

    elif model_name == "llama":

        gptconf = LLaMAConfig(
            block_size=1024, 
            vocab_size=vocab_size,
            padded_vocab_size=vocab_size,
            n_layer=12,
            n_head=12,
            n_embd=768
        )
        model = LLaMA(gptconf)

    else:
        raise NotImplementedError

    return model


def codes_to_tokens(
    labels: list, 
    codes: np.ndarray, 
    tokenizer: BaseTokenizer
) -> np.ndarray:

    B, L = codes.shape
    mat = []

    for i in range(B):
        tmp = [tokenizer.stoi("<sos>")]
        tmp.extend([tokenizer.stoi(labels[i])])
        tmp.extend([tokenizer.stoi(s) for s in codes[i]])
        tmp.extend([tokenizer.stoi("<eos>")])

        mat.append(tmp)

    output = np.array(mat)

    return output


def validate(
    dataset: Dataset,
    sr: int,
    clip_duration: float,
    batch_size: int, 
    codec_model: object,
    n_quantizers: int,
    tokenizer: BaseTokenizer,
    model: nn.Module, 
    evaluate_num: int
) -> float:
    r"""Calculate SDR.
    """

    clip_samples = round(clip_duration * sr)
    losses = []

    for n in tqdm(range(evaluate_num)):

        label = dataset[n]["label"]
        audio_path = dataset[n]["audio_path"]
        duration = librosa.get_duration(path=audio_path)

        audio = load(path=audio_path, sr=sr)

        loss = forward_in_batch(
            label=label,
            audio=audio, 
            codec_model=codec_model, 
            n_quantizers=n_quantizers,
            tokenizer=tokenizer, 
            model=model, 
            clip_samples=clip_samples, 
            batch_size=batch_size
        )
        
        losses.append(loss)

    return np.mean(losses)


def forward_in_batch(
    label: str,
    audio: np.ndarray, 
    codec_model: object, 
    n_quantizers: int,
    tokenizer: BaseTokenizer, 
    model: nn.Module, 
    clip_samples: int, 
    batch_size: int
) -> float:

    device = next(model.parameters()).device

    audio_samples = audio.shape[1]
    padded_audio_samples = round(np.ceil(audio_samples / clip_samples) * clip_samples)
    audio = librosa.util.fix_length(data=audio, size=padded_audio_samples, axis=-1)

    clips = librosa.util.frame(
        audio, 
        frame_length=clip_samples, 
        hop_length=clip_samples
    )
    # shape: (channels_num, clip_samples, clips_num)
    
    clips = clips.transpose(2, 0, 1)
    # shape: (clips_num, channels_num, clip_samples)

    clips_num = clips.shape[0]

    pointer = 0
    losses = []

    while pointer < clips_num:

        batch_clips = torch.Tensor(clips[pointer : pointer + batch_size].copy()).to(device)
        batch_labels = [label] * len(batch_clips)

        with torch.no_grad():
            
            batch_codes, _ = CodecWrapper.encode(
                audio=batch_clips, 
                model=codec_model, 
                n_quantizers=n_quantizers
            )

            # Codes to tokens
            tokens = codes_to_tokens(
                labels=batch_labels, 
                codes=batch_codes, 
                tokenizer=tokenizer
            )

            tokens = torch.LongTensor(tokens).to(device)
            input_tokens = tokens[:, 0 : -1]
            target_tokens = tokens[:, 1 :].contiguous()
            
            model.eval()
            logits, loss = model(
                idx=input_tokens, 
                targets=target_tokens
            )

        losses.append(loss.item())
        pointer += batch_size

    return np.mean(losses)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="llama")
    args = parser.parse_args()

    train(args)