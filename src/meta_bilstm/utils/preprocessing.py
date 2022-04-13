from sklearn.model_selection import train_test_split
import conllu


class Preprocessor:

    def __init__(self, test_size=0.25):
        self.test_size = test_size

    def convert_to_dataset(self, train_sentences, test_sentences):
        tags = self.get_pos_tags(train_sentences)
        tag_to_idx = {tag: i for i, tag in enumerate(tags)}
        idx_to_tag = {i: tag for i, tag in enumerate(tags)}

        train_list_dataset = []
        chars = set()

        for sentence in train_sentences:
            cur_sent = []
            for word in sentence:
                preprocessed_word = self.preprocess_text(word['form'])
                chars = chars.union(set(preprocessed_word))
                cur_sent.append((preprocessed_word, tag_to_idx[word['upos']]))
            train_list_dataset.append(cur_sent)

        test_list_dataset = []

        for sentence in test_sentences:
            cur_sent = []
            for word in sentence:
                preprocessed_word = self.preprocess_text(word['form'])
                chars = chars.union(set(preprocessed_word))
                cur_sent.append((preprocessed_word, tag_to_idx[word['upos']]))
            test_list_dataset.append(cur_sent)

        char_to_idx = {char: idx + 3 for idx, char in enumerate(chars)}
        idx_to_char = {idx + 3: char for idx, char in enumerate(chars)}

        char_to_idx['<pad>'] = 0
        char_to_idx['|'] = 1
        char_to_idx["unknown"] = 2
        idx_to_char[0] = '<pad>'
        idx_to_char[1] = '|'
        idx_to_char[2] = "unknown"

        if test_sentences is None:
            train, test = train_test_split(train_list_dataset, test_size=self.test_size)
            return {
                "datasets": {
                    "train": train,
                    "test": test,
                },
                "tags": {
                    "tag_to_idx": tag_to_idx,
                    "idx_to_tag": idx_to_tag,
                },
                "chars": {
                    "char_to_idx": char_to_idx,
                    "idx_to_char": idx_to_char
                }
            }
        else:
            return {
                "datasets": {
                    "train": train_list_dataset,
                    "test": test_list_dataset,
                },
                "tags": {
                    "tag_to_idx": tag_to_idx,
                    "idx_to_tag": idx_to_tag,
                },
                "chars": {
                    "char_to_idx": char_to_idx,
                    "idx_to_char": idx_to_char
                }
            }

    def get_train_test_list_datasets(self, train_path, test_path):
        with open(train_path, 'r') as f:
            train_sentences = conllu.parse(f.read())
        with open(test_path, 'r') as f:
            test_sentences = conllu.parse(f.read())
        return self.convert_to_dataset(train_sentences, test_sentences)

    @staticmethod
    def get_pos_tags(sentences):
        tags = set()
        for sentence in sentences:
            for word in sentence:
                tags.add(word['upos'])
        return tags

    @staticmethod
    def preprocess_text(string):
        string = string.lower()
        return string
