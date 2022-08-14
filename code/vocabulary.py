import json


class Vocabulary:

    def __init__(self):
        self.unique_words = []
        self.word2index = {"OOV": 0}
        self.word2count = {}
        self.n_words = 1  # OOV = out-of-vocab

    def add_text(self, sentence):
        for word in sentence.split():
            self.add_word(word)

    def add_word(self, word):
        if word not in self.unique_words:
            self.unique_words.append(word)
            self.word2count[word] = 1
        else:
            self.word2count[word] += 1

    def remove_rare(self, min_count=2):
        for word in self.unique_words:
            if self.word2count[word] < min_count:
                self.unique_words.remove(word)
                del self.word2count[word]

    def index_words(self):
        self.n_words = 1
        for word in self.unique_words:
            self.word2index[word] = self.n_words
            self.n_words += 1

    def tokenize_text(self, text):
        pass

    def get_index(self, tokens):
        index = []
        for token in tokens:
            if token not in self.unique_words:  # then its 'oov'
                token = "OOV"
            index.append(self.word2index[token])
        return index

    def save(self):
        f = open("my_voc.json", "w")
        json.dump(self.word2index, f)
        f.close()

    def read(self):
        f = open("my_voc.json", "r")
        word2index = json.loads(f.read())
        f.close()
        self.word2index = word2index
        self.n_words = len(word2index.keys())
        self.unique_words = list(word2index.keys())
        self.unique_words.remove('OOV')