from collections import defaultdict

import torch
import torch.nn as nn
from tqdm import tqdm


class WordPretrainedEmbbedings(nn.Module):

    def __init__(self, embeddings):
        super().__init__()
        self.emb_layer = nn.Embedding.from_pretrained(
            embeddings=embeddings,
            freeze=True,
            padding_idx=0
        )

    def forward(self, x):
        return self.emb_layer(x)


class WordTrainableEmbeddings(nn.Module):

    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.emb_layer = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=0,
        )

    def forward(self, x):
        return self.emb_layer(x)


class WordEmbeddingLayer(nn.Module):

    def __init__(self, dataset, pretrained_path):
        super().__init__()
        emb_tensor, token_to_idx, idx_to_token = self.read_embeddings(pretrained_path)
        num_emb, emb_dim = emb_tensor.shape
        self.emb_dim = emb_dim
        self.token_to_idx = token_to_idx
        self.idx_to_token = idx_to_token

        new_words_count = self.update_token_to_idx_vocab(dataset)
        emb_tensor = torch.cat([emb_tensor, *[torch.zeros(1, emb_dim) for _ in range(new_words_count)]])

        self.pretrained_embs = WordPretrainedEmbbedings(emb_tensor)
        self.trainable_embs = WordTrainableEmbeddings(num_emb + new_words_count, emb_dim)

    def update_token_to_idx_vocab(self, dataset):
        dataset_vocab = self.build_vocab(dataset)
        emb_vocab = set(self.token_to_idx.keys())
        intersect_vocab = set.difference(dataset_vocab, emb_vocab)

        shift = len(self.token_to_idx)
        spec_token_to_idx = {
            word: i + shift for i, word in enumerate(intersect_vocab)
        }
        spec_idx_to_token = {
            i + shift: word for i, word in enumerate(intersect_vocab)
        }
        self.token_to_idx = {**self.token_to_idx, **spec_token_to_idx}
        self.idx_to_token = {**self.idx_to_token, **spec_idx_to_token}
        return len(intersect_vocab)

    @staticmethod
    def build_vocab(dataset):
        vocab = set()
        for sentence in dataset:
            for pair in sentence:
                vocab.add(pair[0])
        return vocab

    def forward(self, x):
        return self.trainable_embs(x) + self.pretrained_embs(x)

    @staticmethod
    def read_embeddings(path):
        with open(path) as f:
            embs = []
            for line in f:
                embs.append(line)
        embs = embs[1:]
        token_to_idx = {'<PAD>': 0, '<OOV>': 1}
        idx_to_token = {0: '<PAD>', 1: "<OOV>"}
        i = 2
        list_embs = [torch.FloatTensor([0] * 300), torch.FloatTensor([0] * 300)]
        multis = defaultdict(lambda: 1)
        for line in tqdm(embs):
            word, *vec = line.split(' ')
            word, _ = word.split("_")
            vec = torch.FloatTensor(list(map(float, vec)))
            if word not in token_to_idx:
                token_to_idx[word] = i
                idx_to_token[i] = word
                list_embs.append(vec)
                i += 1
            else:
                multis[token_to_idx[word]] += 1
                list_embs[token_to_idx[word]] += vec
        for idx, divider in multis.items():
            list_embs[idx] /= divider
        return torch.stack(list_embs), token_to_idx, idx_to_token


class WordBiLSTM(nn.Module):

    def __init__(self,
                 hidden_dim,
                 dataset,
                 output_proj_size,
                 device,
                 mlp_proj_size,
                 num_layers,
                 dropout,
                 pretrained_embs_path,
                 ):
        super().__init__()
        self.device = device
        self.emb_layer = WordEmbeddingLayer(dataset, pretrained_embs_path)
        self.rnn = nn.LSTM(
            input_size=self.emb_layer.emb_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.mlp_proj_size = mlp_proj_size
        self.mlp = nn.Linear(2 * hidden_dim, mlp_proj_size)
        self.relu = nn.ReLU()

        self.output_proj = nn.Linear(mlp_proj_size, output_proj_size)

    def forward(self, x):
        inds, lens = x
        inds, lens = inds.to(self.device), lens.to(self.device)
        embedded = self.emb_layer(inds)
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths=lens,
            batch_first=True,
            enforce_sorted=False
        )

        output, _ = self.rnn(packed, self.get_initial_state(inds))
        output, lens = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        word_encodings = self.mlp(self.relu(output))
        output = self.output_proj(self.relu(word_encodings))
        return output, word_encodings, lens

    def get_initial_state(self, inp):
        shape = self.rnn.get_expected_hidden_size(inp, None)
        return torch.zeros(shape).to(self.device), torch.zeros(shape).to(self.device)
