from datetime import datetime
import pandas as pd
import re
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_sequence

# parameters
batch_size = 10
max_tweet_length = 40
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
embedding_dim = 128
hidden_dim = 256
num_of_hidden_layers = 2
num_classes = 3
learning_rate = 0.005
dropout_rate = 0.5
num_epochs = 20
min_count_words = 2


# vocabulary class
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


# def collate_fn_pad_fixed(batch):
#     """ Align lengths according to max_tweet_length"""
#     X, Y = list(zip(*batch))
#     X = [sample + ([0] * (max_tweet_length - len(sample)))
#          if len(sample) < max_tweet_length
#          else sample[:max_tweet_length] for sample in X]
#     return torch.tensor(X, dtype=torch.int32), torch.tensor(Y)


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


start_time = datetime.now()
print('\n=====PROCESSING DATA=====')
# pre-process data
# load data
train_df = pd.read_csv('data/trainEmotions.csv')
test_df = pd.read_csv('data/testEmotions.csv')
# normalize text
train_df['normalize'] = train_df['content'].apply(normalize_string)
test_df['normalize'] = test_df['content'].apply(normalize_string)
# convert emotions to int labels
emotion_to_labels = {"happiness": 0, "neutral": 1, "sadness": 2}
train_df["labels"] = train_df["emotion"].apply(lambda x: emotion_to_labels[x])
test_df["labels"] = test_df["emotion"].apply(lambda x: emotion_to_labels[x])

# split train data to train and validation
X_train, X_val, y_train, y_val = train_test_split(
    train_df['normalize'].values, train_df['labels'].values,
    test_size=0.2, random_state=27
)

# create vocabulary
vocab = Vocabulary()
for s in X_train:
    vocab.add_text(s)
vocab.remove_rare(min_count_words)  # for regularization. remove rare words.
vocab.index_words()

# create datasets
train_dataset = MyDataset(X_train, y_train, vocab)
val_dataset = MyDataset(X_val, y_val, vocab)
test_dataset = MyDataset(test_df['normalize'].values,
                         test_df['labels'].values, vocab)
# create data loaders
train_loader = DataLoader(train_dataset, batch_size, shuffle=True,
                          num_workers=0, collate_fn=collate_fn_pad)
val_loader = DataLoader(val_dataset, batch_size, shuffle=False,
                        num_workers=0, collate_fn=collate_fn_pad)
test_loader = DataLoader(test_dataset, batch_size, shuffle=False,
                         num_workers=0, collate_fn=collate_fn_pad)

print(f'Time elapsed: {(datetime.now() - start_time)}')


# create the model
class RNN(torch.nn.Module):

    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super().__init__()

        # self.vocabulary = vocab
        self.embedding = torch.nn.Embedding(input_dim,
                                            embedding_dim)
        self.rnn = torch.nn.LSTM(embedding_dim,
                                 hidden_dim,
                                 num_layers=num_of_hidden_layers,
                                 batch_first=True)

        # self.rnn = torch.nn.RNN(embedding_dim,
        #                         hidden_dim,
        #                         batch_first=True,
        #                         nonlinearity='relu')

        self.dropout = torch.nn.Dropout(dropout_rate)
        # self.fc = torch.nn.Linear(hidden_dim, output_dim)
        self.linear_layers = torch.nn.Sequential(
            torch.nn.Linear(in_features=hidden_dim,
                            out_features=hidden_dim * 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(in_features=hidden_dim * 2,
                            out_features=output_dim))

    def forward(self, index):
        # index dim: [batch size, num_of_tokens]
        embedded = self.embedding(index)
        # embedded dim: [batch size, sentence length, embedding dim]

        output, (hidden, cell) = self.rnn(embedded)  # when lstm
        # output, hidden = self.rnn(embedded)  # when rnn
        # output dim: [ batch size, sentence length, hidden dim]
        # hidden dim: [num_layers, batch size, hidden dim]

        # hidden = self.dropout(hidden)
        # output = self.fc(hidden[-1, :, :])  # feed forward last hidden layer
        output = self.linear_layers(hidden[-1, :, :])
        return output


model = RNN(input_dim=vocab.n_words,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=num_classes,
            )

model = model.to(device)
# load saved model
# model.load_state_dict(torch.load('model_22-06-02-00:03:06.pkl'))  #only after strart of training
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()


# Training
def compute_accuracy(model, data_loader, device):
    with torch.no_grad():
        correct_pred, num_examples = 0, 0

        for i, (features, targets) in enumerate(data_loader):
            features = features.to(device)
            targets = targets.float().to(device)

            outputs = model(features)
            _, predicted_labels = torch.max(outputs, 1)

            num_examples += targets.size(0)
            correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float().item() / num_examples * 100


train_accuracy = []
val_accuracy = []
test_accuracy = []


print('\n=====START TRAINING=====')
start_time = datetime.now()

for epoch in range(num_epochs):
    model.train()
    for batch_idx, (text, labels) in enumerate(train_loader):

        text = text.to(device)
        labels = labels.to(device)

        # FORWARD AND BACK PROP
        outputs = model(text)
        # print("----output shape----")
        # print(outputs.shape)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()

        loss.backward()

        # UPDATE MODEL PARAMETERS
        optimizer.step()

        # LOGGING
        if not batch_idx % 100:
            print(f'Epoch: {epoch + 1:03d}/{num_epochs:03d} | '
                  f'Batch {batch_idx:03d}/{len(train_loader):03d} | '
                  f'Loss: {loss:.4f}')

    with torch.set_grad_enabled(False):
        train_acc = compute_accuracy(model, train_loader, device)
        train_accuracy.append(train_acc)
        val_acc = compute_accuracy(model, val_loader, device)
        val_accuracy.append(val_acc)
        print(f'training accuracy: {train_acc:.2f}%'
              f'\nvalid accuracy: {val_acc:.2f}%')

    print(f'Time elapsed: {(datetime.now() - start_time)}')

    # save model and results every 10 epochs
    if not (epoch + 1) % 10:
        timestamp = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
        torch.save(model.state_dict(),
                   f'lstm_model_{num_of_hidden_layers}_hidden_{timestamp}_{epoch + 1}-epochs.pkl')
        data = {"train_acc": train_accuracy,
                "val_acc": val_accuracy}

        res = pd.DataFrame(data)
        res.to_csv(f'results_lstm_{num_of_hidden_layers}_hidden_2_linear.csv')

print(f'Total Training Time: {(datetime.now() - start_time)}')
print(f'Test accuracy: {compute_accuracy(model, test_loader, device):.2f}%')
