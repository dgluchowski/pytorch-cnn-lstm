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

Measured test accuracy (trained on a Colab T4 GPU):

| Task | Model | Test accuracy |
|---|---|---|
| MNIST digit classification | CNN, 2 conv blocks, 10 epochs | **99.0%** |
| CIFAR-10 image classification | CNN, 3 conv blocks + augmentation + dropout, 30 epochs | **81.3%** |
| SST 3-class sentiment | bidirectional LSTM, best-validation checkpoint | **62.0%** |

The sentiment model overfits after a handful of epochs (training accuracy keeps
rising while validation plateaus around 60%), so the reported figure uses the
lowest-validation-loss checkpoint. Each script prints its own per-epoch curves and
final test accuracy; re-running reproduces these within about a point.

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
