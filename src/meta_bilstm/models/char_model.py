import torch
import torch.nn as nn


class CharEmbeddings(nn.Module):

    def __init__(self, emb_dim, num_embs):
        super().__init__()
        self.emb_dim = emb_dim
        self.emb_layer = nn.Embedding(
            num_embeddings=num_embs,
            embedding_dim=emb_dim,
            padding_idx=0,
        )

    def forward(self, x):
        return self.emb_layer(x)


class CharBiLSTM(nn.Module):

    def __init__(self,
                 emb_dim,
                 hidden_dim,
                 output_proj_size,
                 num_embs,
                 device,
                 mlp_proj_size,
                 num_layers,
                 dropout):
        super().__init__()
        self.device = device
        self.emb_layer = CharEmbeddings(emb_dim, num_embs)
        self.rnn = nn.LSTM(
            input_size=self.emb_layer.emb_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.mlp = nn.Linear(4 * hidden_dim, mlp_proj_size)
        self.mlp_proj_size = mlp_proj_size
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

        gathered = self.get_word_encodings(output, x)
        padded_words = nn.utils.rnn.pad_sequence(gathered, batch_first=True)

        word_encodings = self.mlp(self.relu(padded_words))
        output = self.output_proj(self.relu(word_encodings))
        return output, word_encodings, lens

    def get_word_encodings(self, output, inp):
        word_borders = self.get_words_borders(inp)
        result = []
        for i in range(len(output)):
            result.append(
                torch.cat(
                    [output[i][word_borders[i]][::2], output[i][word_borders[i]][1::2]],
                    dim=1
                )
            )
        return result

    @staticmethod
    def get_words_borders(inp):
        result = []
        ind_tensor, lens = inp
        for row, cur_len in zip(ind_tensor, lens):
            j = 0
            cur_inds = []
            start_ind = 0
            while j < cur_len and row[j] != 0:
                if row[j] == 1:
                    cur_inds = cur_inds + [start_ind, j - 1]
                    start_ind = j + 1
                j += 1
            else:
                cur_inds = cur_inds + [start_ind, j - 1]
            result.append(torch.LongTensor(cur_inds))
        return result

    def get_initial_state(self, inp):
        shape = self.rnn.get_expected_hidden_size(inp, None)
        return torch.zeros(shape).to(self.device), torch.zeros(shape).to(self.device)