"""Sentiment classification on the Stanford Sentiment Treebank (SST) with a
bidirectional LSTM (PyTorch, no torchtext).

Downloads SST, builds a vocabulary from the training split, and trains a
3-layer bidirectional LSTM with dropout to classify sentences as negative /
neutral / positive. Uses a padded-sequence DataLoader and keeps the
lowest-validation-loss weights.

Run:
    python sentiment_lstm/sentiment_lstm.py

SST is downloaded automatically into ../data/sst (git-ignored).

Note: SST keys its sentiment scores by *phrase id*, not sentence index, so each
sentence is mapped to its phrase id through `dictionary.txt` (an exact text match)
before its label is attached. Joining sentence index directly to phrase id would
scramble the labels.
"""

import copy
import io
import os
import re
import zipfile
from collections import Counter

import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "sst")
SST_URL = "http://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip"


def download_sst(dest: str) -> str:
    """Download and extract SST once; return the path to its data folder."""
    root = os.path.join(dest, "stanfordSentimentTreebank")
    if not os.path.isdir(root):
        os.makedirs(dest, exist_ok=True)
        print("Downloading SST...")
        response = requests.get(SST_URL)
        response.raise_for_status()
        zipfile.ZipFile(io.BytesIO(response.content)).extractall(dest)
    return root


sst_root = download_sst(DATA_DIR)

# Load the SST files. Sentiment scores are keyed by phrase id, so map each
# sentence to its phrase id via dictionary.txt (exact text match), then attach
# the score. Finally join the train/val/test split assignment.
sentences = pd.read_csv(os.path.join(sst_root, "datasetSentences.txt"), sep="\t")
dictionary = pd.read_csv(os.path.join(sst_root, "dictionary.txt"), sep="|",
                         header=None, names=["phrase", "phrase_id"])
labels = pd.read_csv(os.path.join(sst_root, "sentiment_labels.txt"), sep="|")
splits = pd.read_csv(os.path.join(sst_root, "datasetSplit.txt"), sep=",")

data = sentences.merge(dictionary, left_on="sentence", right_on="phrase", how="inner")
data = data.merge(labels, left_on="phrase_id", right_on="phrase ids", how="inner")
data = data.merge(splits, on="sentence_index", how="inner")


# Group fine-grained scores (0-1) into 3 classes.
def get_label(score):
    if score <= 0.4:
        return 0  # negative
    elif score <= 0.6:
        return 1  # neutral
    else:
        return 2  # positive


data["label"] = data["sentiment values"].apply(get_label)

# splitset_label: 1 = train, 2 = test, 3 = val
train_data = data[data["splitset_label"] == 1]
val_data = data[data["splitset_label"] == 3]
test_data = data[data["splitset_label"] == 2]


def tokenizer(text):
    text = re.sub(r"[^A-Za-z0-9 ]+", "", text).lower()
    return text.split()


# Build vocabulary from the training split.
word_counts = Counter()
for sentence in train_data["sentence"]:
    word_counts.update(tokenizer(sentence))
vocab = ["<unk>", "<pad>"] + list(word_counts.keys())
word_to_idx = {word: i for i, word in enumerate(vocab)}

padding_idx = 1
vocab_size = len(vocab)
label_size = 3
embedding_dim = 300
hidden_dim = 512


def text_to_tensor(text):
    return torch.tensor([word_to_idx.get(word, 0) for word in tokenizer(text)])  # 0 = <unk>


class SSTDataset(Dataset):
    def __init__(self, df):
        self.texts = df["sentence"].tolist()
        self.labels = df["label"].tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return text_to_tensor(self.texts[idx]), torch.tensor(self.labels[idx])


def collate_batch(batch):
    texts, labels = zip(*batch)
    texts = pad_sequence(texts, padding_value=padding_idx, batch_first=True)
    labels = torch.stack(labels)
    return texts.to(device), labels.to(device)


train_iter = DataLoader(SSTDataset(train_data), batch_size=32, shuffle=True, collate_fn=collate_batch)
val_iter = DataLoader(SSTDataset(val_data), batch_size=32, shuffle=False, collate_fn=collate_batch)
test_iter = DataLoader(SSTDataset(test_data), batch_size=32, shuffle=False, collate_fn=collate_batch)


def run_epoch(model, iterator, criterion, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    epoch_loss, epoch_acc = 0.0, 0.0
    with torch.set_grad_enabled(training):
        for texts, labels in iterator:
            if training:
                optimizer.zero_grad()
            predictions = model(texts)
            loss = criterion(predictions, labels)
            if training:
                loss.backward()
                optimizer.step()
            acc = (predictions.argmax(1) == labels).float().mean()
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, label_size, padding_idx):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.rnn = nn.LSTM(
            embedding_dim, hidden_dim, num_layers=3, bidirectional=True,
            dropout=0.5, batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim * 2, label_size)
        self.dropout = nn.Dropout(0.6)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        _, (hidden, _) = self.rnn(embedded)
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        return self.fc(hidden)


if __name__ == "__main__":
    model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, label_size, padding_idx).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss().to(device)

    best_val_loss = float("inf")
    best_state = None
    num_epochs = 20

    for epoch in range(num_epochs):
        train_loss, train_acc = run_epoch(model, train_iter, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_iter, criterion)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
        print(f"Epoch: {epoch+1:02}")
        print(f"\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%")
        print(f"\t Val. Loss: {val_loss:.3f} |  Val. Acc: {val_acc*100:.2f}%")

    model.load_state_dict(best_state)
    test_loss, test_acc = run_epoch(model, test_iter, criterion)
    print(f"\nBest-model Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%")
