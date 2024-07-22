import numpy as np
import torch

from einops import rearrange
import dac


class CodecWrapper:
    def __init__():
        pass

    @staticmethod
    def load_model():
        model_path = dac.utils.download(model_type="44khz")
        model = dac.DAC.load(model_path)
        return model

    @staticmethod
    def encode(audio, model, n_quantizers=None):

        with torch.no_grad():
            model.eval()
            z, codes, latents, _, _ = model.encode(audio, n_quantizers=n_quantizers)

        # if n_quantizers:
        #     codes = codes[:, 0 : n_quantizers, :]

        codes = rearrange(codes, 'b k t -> b (t k)').cpu().numpy()
        # (B, T x K)

        return codes, latents

    @staticmethod
    def decode_from_codes(codes, model, n_quantizers):
        
        codes = rearrange(codes, 'b (t k) -> b k t', k=n_quantizers)

        with torch.no_grad():
            model.eval()
            z, _, _ = model.quantizer.from_codes(codes)
            audio = model.decode(z)

        return audio

    @staticmethod
    def decode_from_latents(latents, model):

        with torch.no_grad():
            model.eval()        
            z, _, _ = model.quantizer.from_latents(latents)
            audio = model.decode(z)

        return audio