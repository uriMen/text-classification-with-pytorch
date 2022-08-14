import argparse
from my_models import RNN, Transformer
from vocabulary import Vocabulary
from datetime import datetime
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import re
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_sequence

# parameters
batch_size = 8
max_tweet_length = 280 #40
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
embedding_dim = 128
hidden_dim = 256
num_of_hidden_layers = 2
num_classes = 3
learning_rate = 0.005
dropout_rate = 0.5
num_epochs = 20
min_count_words = 2
# transformer params
num_of_blocks = 2
forward_expansion = 4
num_of_heads = 8


class MyDataset(Dataset):
    def __init__(self, X, y, voc: Vocabulary):
        self.voc = voc
        self.X = self.text_to_index(X)
        self.y = y

    def text_to_index(self, X):
        index = []
        for x in X:
            tokens = x.split()
            index.append(self.voc.get_index(tokens))
        return index

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def collate_fn_pad_fixed(batch):
    """ Align lengths according to max_tweet_length"""
    X, Y = list(zip(*batch))
    X = [sample + ([0] * (max_tweet_length - len(sample)))
         if len(sample) < max_tweet_length
         else sample[:max_tweet_length] for sample in X]
    return torch.tensor(X, dtype=torch.int32), torch.tensor(Y)


def collate_fn_pad(batch):
    """ Align lengths according to longest in batch"""
    X, Y = list(zip(*batch))
    # get sequence lengths
    lengths = torch.tensor([len(t) for t in X]).to(device)
    # pad
    batch = [torch.tensor(t).to(device) for t in X]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True)
    labels = torch.tensor(Y)
    return batch, labels


def normalize_string(s):
    s = s.lower().strip()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?0-9]+", r" ", s)
    return s


if __name__ == "__main__":

    # start_time = datetime.now()
    saved_model_name = "model.pkl"

    # Parsing script arguments
    parser = argparse.ArgumentParser(description='Process input')
    parser.add_argument('input_file', type=str, help='Input file path')
    args = parser.parse_args()

    # pre-process data
    # load data
    # train_df = pd.read_csv('data/trainEmotions.csv')
    test_df = pd.read_csv(args.input_file)
    # normalize text
    test_df['normalize'] = test_df['content'].apply(normalize_string)
    # convert emotions to int labels
    emotion_to_labels = {"happiness": 0, "neutral": 1, "sadness": 2}
    labels_to_emotion = {0: "happiness", 1: "neutral", 2: "sadness"}

    test_df["labels"] = test_df["emotion"].apply(lambda x: emotion_to_labels[x])

    # load vocabulary
    vocab = Vocabulary()
    vocab.read()

    # create datasets
    test_dataset = MyDataset(test_df['normalize'].values,
                             test_df['labels'].values, vocab)
    # create data loaders
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False,
                             num_workers=0, collate_fn=collate_fn_pad)

    # print(f'Time elapsed: {(datetime.now() - start_time)}')

    # create the model
    model = Transformer(vocab_size=vocab.n_words,
                        embed_size=embedding_dim,
                        num_layers=num_of_blocks,
                        heads=num_of_heads,
                        device=device,
                        forward_expansion=forward_expansion,
                        dropout=dropout_rate,
                        max_length=max_tweet_length,
                        num_classes=num_classes,
                        pad_idx=0)

    model = model.to(device)
    # load saved model
    model.load_state_dict(torch.load(saved_model_name, map_location=device))

    predictions = []

    for i, (features, targets) in enumerate(test_loader):
        features = features.to(device)
        targets = targets.float().to(device)

        outputs = model(features)
        _, predicted_labels = torch.max(outputs, 1)
        predictions += predicted_labels

    emotions = np.array([labels_to_emotion[i.item()] for i in predictions])
    test_df['predicted'] = emotions
    # test_df.to_csv("with_predict.csv")
    content = test_df["content"].values
    data = {"emotion": emotions, "content": content}
    prediction_df = pd.DataFrame(data)
    prediction_df.to_csv("prediction.csv", index=False)
