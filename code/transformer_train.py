from vocabulary import Vocabulary
from my_models import Transformer, RNN
from datetime import datetime
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

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
# X_train, X_val, y_train, y_val = train_test_split(
#     train_df['normalize'].values, train_df['labels'].values,
#     test_size=0.2, random_state=27
# )

X_train = train_df['normalize'].values
y_train = train_df['labels'].values

# create vocabulary
vocab = Vocabulary()
for s in X_train:
    vocab.add_text(s)
vocab.remove_rare(min_count_words)  # for regularization. remove rare words.
vocab.index_words()
vocab.save()

# create datasets
train_dataset = MyDataset(X_train, y_train, vocab)
# val_dataset = MyDataset(X_val, y_val, vocab)
test_dataset = MyDataset(test_df['normalize'].values,
                         test_df['labels'].values, vocab)
# create data loaders
train_loader = DataLoader(train_dataset, batch_size, shuffle=True,
                          num_workers=0, collate_fn=collate_fn_pad)
# val_loader = DataLoader(val_dataset, batch_size, shuffle=False,
#                         num_workers=0, collate_fn=collate_fn_pad)
test_loader = DataLoader(test_dataset, batch_size, shuffle=False,
                         num_workers=0, collate_fn=collate_fn_pad)

print(f'Time elapsed: {(datetime.now() - start_time)}')


# create the transformer
print("\n=====CREATING THE MODEL=====")
start_time = datetime.now()

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

# model = RNN(input_dim=vocab.n_words,
#             embedding_dim=embedding_dim,
#             hidden_dim=hidden_dim,
#             output_dim=num_classes,
#             num_layers=num_of_hidden_layers,
#             dropout_rate=dropout_rate
#             )

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                 milestones=[5, 10, 15],
                                                 gamma=0.2)
criterion = torch.nn.CrossEntropyLoss()
num_of_params = sum(param.numel() for param in model.parameters())
print(f'Num of params: {num_of_params}')
print(f'Time elapsed: {(datetime.now() - start_time)}')


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


def compute_loss(model, data_loader, device):
    with torch.no_grad():
        loss_ = []
        for i, (features, labels) in enumerate(data_loader):
            features = features.to(device)
            labels = labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss_.append(float(loss.item()))
    return np.mean(loss_)


train_accuracy = []
val_accuracy = []
test_accuracy = []

train_loss = []
test_loss = []


print('\n=====START TRAINING=====')
start_time = datetime.now()

for epoch in range(num_epochs):
    model.train()
    for batch_idx, (text, labels) in enumerate(train_loader):

        text = text.to(device)
        labels = labels.to(device)

        # FORWARD AND BACK PROP
        outputs = model(text)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()

        loss.backward()

        # UPDATE MODEL PARAMETERS
        optimizer.step()
        scheduler.step()

        # LOGGING
        if not batch_idx % 100:
            print(f'Epoch: {epoch + 1:03d}/{num_epochs:03d} | '
                  f'Batch {batch_idx:03d}/{len(train_loader):03d} | '
                  f'Loss: {loss:.4f}')

    with torch.set_grad_enabled(False):
        train_acc = compute_accuracy(model, train_loader, device)
        train_accuracy.append(train_acc)
        train_loss.append(compute_loss(model, train_loader, device))
        val_acc = compute_accuracy(model, test_loader, device)
        val_accuracy.append(val_acc)
        test_loss.append(compute_loss(model, test_loader, device))
        print(f'training accuracy: '
              f'{train_acc:.2f}%'
              f'\nvalid accuracy: '
              f'{val_acc:.2f}%')
        # print(f'Test accuracy: '
        #       f'{compute_accuracy(model, test_loader, device):.2f}%')

    print(f'Time elapsed: {(datetime.now() - start_time)}')

    # save model and results every 10 epochs
    if not (epoch + 1) % 10:
        timestamp = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
        torch.save(model.state_dict(),
                   f'transformer_{timestamp}_{epoch + 1}-epochs.pkl')
        data = {"train_acc": train_accuracy,
                "val_acc": val_accuracy,
                "train_loss": train_loss,
                "val_loss": test_loss}

        res = pd.DataFrame(data)
        res.to_csv(f'results_transformer.csv')

print(f'Total Training Time: {(datetime.now() - start_time)}')
print(f'Test accuracy: {compute_accuracy(model, test_loader, device):.2f}%')
