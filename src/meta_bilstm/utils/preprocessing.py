from sklearn.model_selection import train_test_split
import conllu


class Preprocessor:

    def __init__(self, test_size=0.25):
        self.test_size = test_size

    def convert_to_dataset(self, sentences):
        tags = self.get_pos_tags(sentences)
        tag_to_idx = {tag: i for i, tag in enumerate(tags)}
        idx_to_tag = {i: tag for i, tag in enumerate(tags)}

        list_dataset = []
        chars = set()

        for sentence in sentences:
            cur_sent = []
            for word in sentence:
                preprocessed_word = self.preprocess_text(word['form'])
                chars = chars.union(set(preprocessed_word))
                cur_sent.append((preprocessed_word, tag_to_idx[word['upos']]))
            list_dataset.append(cur_sent)

        char_to_idx = {char: idx + 2 for idx, char in enumerate(chars)}
        idx_to_char = {idx + 2: char for idx, char in enumerate(chars)}

        char_to_idx['<pad>'] = 0
        char_to_idx['|'] = 1
        idx_to_char[0] = '<pad>'
        idx_to_char[1] = '|'

        train, test = train_test_split(list_dataset, test_size=self.test_size)
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

    def get_train_test_list_datasets(self, path):
        with open(path, 'r') as f:
            sentences = conllu.parse(f.read())
        return self.convert_to_dataset(sentences)

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
