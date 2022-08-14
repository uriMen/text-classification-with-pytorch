import torch
import torch.nn as nn


class RNN(torch.nn.Module):

    def __init__(self, input_dim, embedding_dim, hidden_dim,
                 output_dim, num_layers, dropout_rate):
        super().__init__()

        # self.vocabulary = vocab
        self.embedding = torch.nn.Embedding(input_dim,
                                            embedding_dim)
        self.rnn = torch.nn.LSTM(embedding_dim,
                                 hidden_dim,
                                 num_layers=num_layers,
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


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads=8):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        # self.mask = mask
        self.head_dim = embed_size // heads

        # verify dimensions
        assert (self.head_dim * heads == embed_size), "Embed size must be " \
                                                      "divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(self.heads * self.head_dim, self.embed_size)

    def forward(self, values, keys, queries, mask):
        b = queries.shape[0]  # batch size
        v_len, k_len, q_len = values.shape[1], keys.shape[1], queries.shape[1]

        # split embedding into self.head pieces
        values = values.reshape(b, v_len, self.heads, self.head_dim)
        keys = values.reshape(b, k_len, self.heads, self.head_dim)
        queries = values.reshape(b, q_len, self.heads, self.head_dim)

        # Einsum does matrix mult. for query*keys for each training example
        # with every other training example.
        energy = torch.einsum("bqhd,bkhd->bhqk", [queries, keys])
        # queries shape: (b, q_len, heads, heads_dim),
        # keys shape: (b, k_len, heads, heads_dim)
        # energy: (b, heads, q_len, k_len)

        # Mask padded indices so their weights become 0
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # Normalize energy values, so that they sum to 1.
        # Also divide by scaling factor for better stability
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        # attention shape: (b, heads, q_len, k_len)

        out = torch.einsum("bhql,blhd->bqhd", [attention, values]).reshape(
            b, q_len, self.heads * self.head_dim)
        # attention shape: (b, heads, q_len, k_len)
        # values shape: (b, v_len, heads, heads_dim)
        # out after matrix multiply: (b, q_len, heads, head_dim), then
        # we reshape and flatten the last two dimensions.
        # note k_len == v_len

        out = self.fc_out(out)
        # Linear layer doesn't modify the shape, final shape will be
        # (N, query_len, embed_size)

        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)

        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class Transformer(nn.Module):
    def __init__(
            self,
            vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length,
            num_classes,
            pad_idx
    ):

        super(Transformer, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(vocab_size, embed_size, max_norm=1)
        self.position_embedding = nn.Embedding(max_length, embed_size,
                                               max_norm=1)
        self.pad_idx = pad_idx

        self.tblocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size=self.embed_size,
                    heads=heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion)
                for _ in range(num_layers)
            ]
        )

        # Maps the final output sequence to class logits
        self.fc = nn.Linear(embed_size, num_classes)

    def make_mask(self, src):
        mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        # (b, 1, 1, src_len)
        return mask.to(self.device)

    def forward(self, x):
        b, seq_length = x.shape
        mask = self.make_mask(x)
        positions = torch.arange(0, seq_length).expand(b, seq_length).\
            to(self.device)
        out = self.word_embedding(x) + self.position_embedding(positions)
        for block in self.tblocks:
            out = block(out, out, out, mask)

        # Average-pool over the t dimension and project to class
        # probabilities
        out = self.fc(out.mean(dim=1))
        return nn.functional.log_softmax(out, dim=1)
