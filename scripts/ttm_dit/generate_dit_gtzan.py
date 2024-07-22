import argparse
import random
import time
from pathlib import Path

import soundfile
import librosa
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from audiotools import AudioSignal
from einops import rearrange
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
from tqdm import tqdm
import dac

wandb.require("core")

from audidata.io import load
from audidata.datasets import GTZAN

from codec import CodecWrapper
from tokenizers import get_gtzan_tokenizer
from train_dit_gtzan import get_model
from mugen.diffusion import create_diffusion


def generate(args):

    # Arguments
    model_name = args.model_name

    sr = 44100
    n_quantizers = 1
    codec_dim = 1024
    device = "cuda"

    genre = "blues"  # "blues" | "classical" | "country" | "disco" | "hiphop" | "jazz" | "metal" | "pop" | "reggae" | "rock"

    checkpoint_path = "/home/qiuqiangkong/my_code_202308-/mugen/checkpoints/train_dit_gtzan_accelerate/dit/step=70000.pth"

    # Codec
    codec_model = CodecWrapper.load_model()
    codec_model.to(device)

    # Tokenizer
    tokenizer = get_gtzan_tokenizer(
        gtzan_labels=GTZAN.labels, 
        codec_labels=range(codec_dim)
    )

    # Load model
    model = get_model(model_name=model_name)
    
    # Load checkpoint    
    model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)

    diffusion = create_diffusion(timestep_respacing="250")

    class_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    n = len(class_labels)
    z = torch.randn(n, 16, 861, device=device)  # (8, 4, 64, 64)
    y = torch.tensor(class_labels, device=device)

    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)    # (8, 4, 64, 64)
    y_null = torch.tensor([10] * n, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=4.0)

    latents = diffusion.p_sample_loop(
        model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )
    
    latents, _ = latents.chunk(2, dim=0)  # Remove null class samples

    # Decode codes to audio
    audio = CodecWrapper.decode_from_latents(
        latents=latents, 
        model=codec_model, 
    )

    Path("results").mkdir(parents=True, exist_ok=True)
    for i in range(len(class_labels)):
        out_path = "results/{}.wav".format(GTZAN.labels[i])
        soundfile.write(file=out_path, data=audio[i, 0].cpu().numpy(), samplerate=sr)
        print("Write out to {}".format(out_path))


def tokens_to_codes(tokens, tokenizer):

    line = tokens[0].cpu().numpy()
    words = [tokenizer.itos(token) for token in line] 
    
    codes = []

    for i in range(2, len(words)):
        
        if words[i] == "<eos>":
            break

        codes.append(words[i])

    return torch.LongTensor(codes)[None, :]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="dit")
    args = parser.parse_args()

    generate(args)