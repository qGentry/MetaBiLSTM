import torch
import torch.nn as nn


class MetaBiLSTM(nn.Module):

    def __init__(self,
                 input_size,
                 hidden_dim,
                 output_proj_size,
                 device,
                 num_layers,
                 dropout):
        super().__init__()
        self.input_size = input_size
        self.rnn = nn.LSTM(
            input_size=self.input_size,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.device = device
        self.relu = nn.ReLU()
        self.output_proj = nn.Linear(2 * hidden_dim, output_proj_size)

    def forward(self, x):
        inds, lens = x
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            inds,
            lengths=lens,
            batch_first=True,
            enforce_sorted=False
        )
        output, _ = self.rnn(packed, self.get_initial_state(inds))
        output, lens = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        output = self.output_proj(self.relu(output))
        meta_model_output = {
            "logits": output,
            "lens": lens,
        }

        return meta_model_output

    def get_initial_state(self, inp):
        shape = self.rnn.get_expected_hidden_size(inp, None)
        return torch.zeros(shape).to(self.device), torch.zeros(shape).to(self.device)
