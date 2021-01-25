import torch


class PosTagDataset(torch.utils.data.Dataset):

    def __init__(self, list_dataset, token_to_idx, char_to_idx):
        self.token_to_idx = token_to_idx
        self.char_to_idx = char_to_idx
        self.data = list_dataset

    def __len__(self):
        return len(self.data)

    @staticmethod
    def prepare_sent(sentence, token_to_idx, char_to_idx):
        labels = []
        word_idx = []
        pure_words = []
        for word, label in sentence:
            labels.append(label)
            pure_words.append(word)
            if word in token_to_idx:
                word_idx.append(token_to_idx[word])
            else:
                word_idx.append(token_to_idx['<OOV>'])
        char_idx = [char_to_idx[char] for char in list('|'.join(pure_words))]
        return {
            "word_idx": torch.LongTensor(word_idx),
            "char_idx": torch.LongTensor(char_idx),
            "labels": torch.LongTensor(labels)
        }

    def __getitem__(self, idx):
        return self.prepare_sent(self.data[idx], self.token_to_idx, self.char_to_idx)


def _collate_fn(data_list):
    word_idx_batch = []
    word_idx_lens = []
    char_idx_batch = []
    char_idx_lens = []
    labels_batch = []
    for data_point in data_list:
        labels_batch.append(data_point['labels'])
        word_idx_batch.append(data_point['word_idx'])
        word_idx_lens.append(len(data_point['word_idx']))
        char_idx_batch.append(data_point['char_idx'])
        char_idx_lens.append(len(data_point['char_idx']))
    return {
        "word_idx": [
            torch.nn.utils.rnn.pad_sequence(word_idx_batch, batch_first=True),
            torch.LongTensor(word_idx_lens),
        ],
        "char_idx": [
            torch.nn.utils.rnn.pad_sequence(char_idx_batch, batch_first=True),
            torch.LongTensor(char_idx_lens),
        ],
        'labels': torch.nn.utils.rnn.pad_sequence(labels_batch, batch_first=True)
    }


def create_dataloader(dataset, batch_size):
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: _collate_fn(x),
    )
    return dataloader
