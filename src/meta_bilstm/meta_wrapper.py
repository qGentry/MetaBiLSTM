import os

import torch

from meta_bilstm.models.char_model import CharBiLSTM
from meta_bilstm.models.word_model import WordBiLSTM
from meta_bilstm.models.meta_model import MetaBiLSTM


class ModelWrapper:

    def __init__(self, params):
        self.meta_model = MetaBiLSTM(**params["meta_model_params"])
        self.word_model = WordBiLSTM(**params["word_model_params"])
        self.char_model = CharBiLSTM(**params["char_model_params"])

    @classmethod
    def load_from_folder(cls, folder="saved_model"):
        wrapper = cls.__new__(cls)
        super(ModelWrapper, wrapper).__init__()
        wrapper.meta_model = torch.load(os.path.join(folder, "meta.pt"))
        wrapper.word_model = torch.load(os.path.join(folder, "word.pt"))
        wrapper.char_model = torch.load(os.path.join(folder, "char.pt"))
        return wrapper

    def save_model(self, folder="saved_model"):
        if not os.path.exists(folder):
            os.mkdir(folder)
        torch.save(self.char_model, os.path.join(folder, "char.pt"))
        torch.save(self.word_model, os.path.join(folder, "word.pt"))
        torch.save(self.meta_model, os.path.join(folder, "meta.pt"))

    def __call__(self, x):
        word_model_logits, word_encodings, word_lens = self.word_model(x['word_idx'])
        word_model_output = {
            "logits": word_model_logits,
            "lens": word_lens
        }
        char_model_logits, char_encodings, char_lens = self.char_model(x['char_idx'])
        char_model_output = {
            "logits": char_model_logits,
            "lens": char_lens,
        }
        word_meta_encodings = torch.cat([char_encodings, word_encodings], dim=2).detach().clone()
        meta_model_output = self.meta_model([word_meta_encodings, word_lens])
        final_output = {
            "meta": meta_model_output,
            "word": word_model_output,
            "char": char_model_output,
        }
        return final_output

    def to(self, device):
        self.char_model = self.char_model.to(device)
        self.word_model = self.word_model.to(device)
        self.meta_model = self.meta_model.to(device)
        return self

    def train(self):
        self.char_model = self.char_model.train()
        self.word_model = self.word_model.train()
        self.meta_model = self.meta_model.train()
        return self

    def eval(self):
        self.char_model = self.char_model.eval()
        self.word_model = self.word_model.eval()
        self.meta_model = self.meta_model.eval()
        return self
