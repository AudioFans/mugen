import numpy as np
import torch

from einops import rearrange
import dac


'''
class CodecWrapper:
    def __init__(self, sr: int, top_rvqs: int):

        self.sr = sr
        self.top_rvqs = top_rvqs

        model_path = dac.utils.download(model_type="44khz")
        self.model = dac.DAC.load(model_path)

    def encode(self, audio):

        with torch.no_grad():
            self.model.eval()
            z, codes, latents, _, _ = self.model.encode(audio)

        codes = codes[:, 0 : self.top_rvqs, :]

        codes = rearrange(codes, 'b k t -> b (t k)').cpu().numpy()
        # (B, T x K)

        return codes

    def decode(self, codes):

        codes = rearrange(codes, 'b (t k) -> b k t', k=self.top_rvqs)

        with torch.no_grad():
            self.model.eval()
            z, _, _ = self.model.quantizer.from_codes(codes)
            audio = self.model.decode(z)

        return audio
'''

'''
class CodecWrapper:
    def __init__(self, sr: int, top_rvqs: int):

        self.sr = sr
        self.top_rvqs = top_rvqs

        model_path = dac.utils.download(model_type="44khz")
        self.model = dac.DAC.load(model_path)

    def encode(self, audio):

        with torch.no_grad():
            self.model.eval()
            z, codes, latents, _, _ = self.model.module.encode(audio)

        codes = codes[:, 0 : self.top_rvqs, :]

        codes = rearrange(codes, 'b k t -> b (t k)').cpu().numpy()
        # (B, T x K)

        return codes

    def decode(self, codes):

        codes = rearrange(codes, 'b (t k) -> b k t', k=self.top_rvqs)

        with torch.no_grad():
            self.model.eval()
            z, _, _ = self.module.quantizer.from_codes(codes)
            audio = self.module.decode(z)

        return audio
'''

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
            z, codes, latents, _, _ = model.encode(audio)

        if n_quantizers:
            codes = codes[:, 0 : n_quantizers, :]

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