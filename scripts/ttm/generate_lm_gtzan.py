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
from train_lm_gtzan import get_model, codes_to_tokens


def generate(args):

    # Arguments
    model_name = args.model_name

    sr = 44100
    n_quantizers = 1
    codec_dim = 1024
    device = "cuda"

    genre = "jazz"  # "blues" | "classical" | "country" | "disco" | "hiphop" | "jazz" | "metal" | "pop" | "reggae" | "rock"

    checkpoint_path = "/home/qiuqiangkong/my_code_202308-/mugen/checkpoints/train_lm_gtzan_accelerate/llama/step=10000.pth"

    # Codec
    codec_model = CodecWrapper.load_model()

    # Tokenizer
    tokenizer = get_gtzan_tokenizer(
        gtzan_labels=GTZAN.labels, 
        codec_labels=range(codec_dim)
    )

    # Load model
    model = get_model(model_name=model_name, vocab_size=tokenizer.vocab_size)
    
    # Load checkpoint    
    model.load_state_dict(torch.load(checkpoint_path))

    model.to(device)

    idx = torch.LongTensor([[
        tokenizer.stoi("<sos>"), 
        tokenizer.stoi(genre)
    ]]).to(device)

    with torch.no_grad():
        model.eval()
        tokens = model.generate(idx, max_new_tokens=1000)

    # Tokens to codes
    codes = tokens_to_codes(tokens, tokenizer)

    # Decode codes to audio
    audio = CodecWrapper.decode_from_codes(
        codes=codes, 
        model=codec_model, 
        n_quantizers=n_quantizers
    )

    Path("results").mkdir(parents=True, exist_ok=True)
    out_path = "results/{}.wav".format(genre)
    soundfile.write(file=out_path, data=audio[0, 0].cpu().numpy(), samplerate=sr)
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
    parser.add_argument('--model_name', type=str, default="llama")
    args = parser.parse_args()

    generate(args)