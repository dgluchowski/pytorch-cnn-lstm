# PyTorch CNN & LSTM Classifiers

Deep-learning models built from scratch in **PyTorch**, spanning **computer
vision** (CNN image classification) and **NLP** (bidirectional-LSTM sentiment
analysis), each with hand-written training/validation loops.

> **Context.** Coursework for **CS 577 (Deep Learning)** at Illinois Institute of
> Technology, cleaned up and consolidated for portfolio use. The datasets are
> downloaded on first run (nothing is committed), and the sentiment model uses a
> plain-PyTorch data pipeline (no deprecated `torchtext`) so it runs on current
> versions.

## Contents

### 1. Image classification — CNNs (`image_classification/`)
- **`mnist_cnn.py`** — a 2-conv-block CNN (32→64 channels, Conv→ReLU→MaxPool) with
  a fully-connected head. Trains with Adam + cross-entropy, tracks train/val
  learning curves (saved to `results/mnist_curves.png`), and reports test accuracy.
- **`cifar10_cnn.py`** — a deeper 3-conv-block CNN (32→64→128) with **dropout** and
  training-time **data augmentation** (random horizontal flip + random crop), a
  clean (un-augmented) test set, and 30 epochs.

### 2. Sentiment analysis — bidirectional LSTM (`sentiment_lstm/`)
- **`sentiment_lstm.py`** — downloads the **Stanford Sentiment Treebank (SST)**,
  builds a vocabulary from the training split, and trains a **3-layer
  bidirectional LSTM** (embedding → BiLSTM → dropout → linear) for 3-class
  sentiment (negative / neutral / positive). Uses a padded-sequence `DataLoader`
  and keeps the lowest-validation-loss weights, then evaluates on the test split.

## What it demonstrates
- **PyTorch fundamentals** implemented by hand: `zero_grad → forward → loss →
  backward → step`, `train()/eval()` modes, `no_grad()`, and manual metric tracking
  (not just `model.fit()`).
- **CNNs** with pooling, dropout, and data augmentation for regularization.
- **Recurrent models** — a bidirectional, multi-layer LSTM with an embedding layer
  and best-model checkpointing.
- Breadth across **vision and NLP**.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python image_classification/mnist_cnn.py      # downloads MNIST, prints test accuracy
python image_classification/cifar10_cnn.py    # downloads CIFAR-10, 30 epochs
python sentiment_lstm/sentiment_lstm.py       # downloads SST, trains the BiLSTM
```

Each script prints per-epoch train/validation metrics and a final **test
accuracy**. A GPU is optional but makes CIFAR-10 and the LSTM much faster (the code
uses CUDA automatically when available).

## Results

Run the scripts to reproduce the numbers on your hardware — each prints its own
train/val curves and final test accuracy. As a rough sanity check, architectures
like these typically land around **~99% on MNIST**, **~75–80% on CIFAR-10** (30
epochs), and, for 3-class SST with a small LSTM, in the **~60–70%** range; treat
these as ballpark references, not measured claims, and use your own run's output.

## Repository layout

```
pytorch-cnn-lstm/
├── image_classification/
│   ├── mnist_cnn.py
│   └── cifar10_cnn.py
├── sentiment_lstm/
│   └── sentiment_lstm.py
├── requirements.txt
├── .gitignore          # datasets, checkpoints, and results are not committed
└── LICENSE
```

## Notes
- **Datasets are not committed.** MNIST/CIFAR-10 come from torchvision and SST is
  downloaded from its Stanford source on first run.
- These are learning-focused implementations on standard benchmarks (MNIST,
  CIFAR-10, SST); the goal is clean, correct PyTorch, not state-of-the-art scores.

## License
[MIT](LICENSE).
