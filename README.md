# 🔥 PyTorch Learning (`pytorch-learning`)

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Google Colab](https://img.shields.io/badge/Colab-T4_GPU-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/)

Hands-on implementations, exercises, and modular deep learning pipelines built while completing Daniel Bourke's **[PyTorch for Deep Learning Bootcamp](https://youtu.be/LyJtbe__2i0)** and companion book (**[learnpytorch.io](https://www.learnpytorch.io/)**).

---

## 🔗 Connecting All the Dots: End-to-End PyTorch Pipeline

```text
[1. Raw Data (CSV / Images)] 
       │
       ▼  (NumPy/Pandas Cleaning ──► torch.utils.data.Dataset & DataLoader)
[2. Batched Tensors (X, y)]  ──► Moved to GPU VRAM via .to(device)
       │
       ▼  (Forward Pass: y_pred = model(X) using nn.Module, @ matmul, & ReLU/Conv2d)
[3. Loss Evaluation]         ──► loss = loss_fn(y_pred, y) (BCEWithLogitsLoss / CrossEntropyLoss)
       │
       ▼  (Backward Pass: loss.backward() computes ∂Loss/∂W via Autograd)
[4. Optimizer Step]          ──► optimizer.step() updates weights (W = W - lr * ∇W)
       │
       ▼  (Evaluation in torch.inference_mode() & Export)
[5. Saved Checkpoint (.pth)] ──► Refactored into Modular CLI Scripts (Ch 05) & Deployed (Ch 09)
```

---

## 📚 Curriculum & Repository Roadmap

| Chapter | Topic | Core PyTorch Modules & Concepts | Status |
| :--- | :--- | :--- | :---: |
| **00** | **PyTorch Fundamentals** | Tensors, `@` / `torch.matmul`, shapes, strides/views, `torch.from_numpy()`, seeds, and `.to("cuda")` GPU execution. | 🔄 In Progress |
| **01** | **PyTorch Workflow** | `nn.Module`, `nn.Parameter`, `nn.Linear`, `L1Loss`/`MSELoss`, `SGD`/`Adam`, the 5-step training loop, and saving/loading `state_dict` (`.pth`). | ⏳ Pending |
| **02** | **Neural Network Classification** | Binary (`BCEWithLogitsLoss`, `torch.sigmoid`) & Multiclass (`CrossEntropyLoss`, `torch.softmax`, `argmax`) classification, and `nn.ReLU` non-linearity. | ⏳ Pending |
| **03** | **Computer Vision & CNNs** | `torchvision.datasets`, `DataLoader` batching, `nn.Conv2d`, `nn.MaxPool2d`, TinyVGG architecture, and `torchmetrics` confusion matrices. | ⏳ Pending |
| **04** | **Custom Datasets** | FoodVision Mini, `ImageFolder`, custom `Dataset` subclasses, `TrivialAugmentWide` data augmentation, and overfitting vs. underfitting diagnostics. | ⏳ Pending |
| **05** | **Going Modular** | Converting `.ipynb` notebooks into production CLI scripts (`data_setup.py`, `model_builder.py`, `engine.py`, `utils.py`, `train.py`). | ⏳ Pending |
| **06–09** | **Advanced (`learnpytorch.io`)** | **06:** Transfer Learning (`weights=...`) • **07:** Experiment Tracking (`TensorBoard`) • **08:** Vision Transformer (ViT) Paper Replication • **09:** Gradio / Hugging Face Deployment. | ⏳ Pending |

---

## 📂 Repository Structure

```text
pytorch-learning/
├── 00_pytorch_fundamentals/       # Tensor operations, GPU setup & NumPy bridge
├── 01_pytorch_workflow/           # End-to-end training/testing loops & saving models
├── 02_pytorch_classification/     # Binary & multiclass classification networks
├── 03_pytorch_computer_vision/    # CNNs (TinyVGG) on FashionMNIST
├── 04_pytorch_custom_datasets/    # Custom Dataset/DataLoader & FoodVision Mini
├── 05_pytorch_going_modular/      # Production Python scripts
│   ├── data_setup.py              # Creates train/test DataLoaders
│   ├── model_builder.py           # TinyVGG nn.Module architecture
│   ├── engine.py                  # train_step(), test_step(), and train() loops
│   ├── utils.py                   # save_model() helper functions
│   └── train.py                   # CLI entry point (python train.py --epochs 10)
├── models/                        # Saved model checkpoints (.pth / .pt) [Git Ignored]
├── data/                          # Downloaded datasets [Git Ignored]
├── .gitignore                     # Excludes Colab artifacts, data/, and large .pth weights
└── README.md
```

---

## ⚡ Quick Reference: Core Loop & Modern PyTorch 2.x

### 1. Device-Agnostic Setup & PyTorch 2.x Speedups
```python
import torch
from torch import nn

# Device-agnostic setup (Colab T4 CUDA, Apple Silicon MPS, or CPU)
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# PyTorch 2.x Upgrades: TensorFloat-32 matmul + Graph Compilation
torch.set_float32_matmul_precision("high")
model = MyModel().to(device)
model = torch.compile(model)
```

### 2. The Standard 5-Step Training & Evaluation Loop
```python
for epoch in range(epochs):
    # --- Training ---
    model.train()
    for X, y in train_dataloader:
        X, y = X.to(device), y.to(device)
        
        y_pred = model(X)                  # 1. Forward pass (outputs raw logits)
        loss = loss_fn(y_pred, y)          # 2. Calculate loss
        optimizer.zero_grad()              # 3. Zero gradients from previous step
        loss.backward()                    # 4. Backpropagation (compute ∂Loss/∂W)
        optimizer.step()                   # 5. Gradient Descent weight update

    # --- Evaluation ---
    model.eval()
    with torch.inference_mode():           # Disables autograd graph tracking to save VRAM
        for X_test, y_test in test_dataloader:
            X_test, y_test = X_test.to(device), y_test.to(device)
            test_pred = model(X_test)
            test_loss = loss_fn(test_pred, y_test)
```

---

## 🛠️ The 3 Golden Rules of PyTorch Debugging

90% of deep learning errors come from three mismatches:
1. **Shape Mismatch:** Inner matrix dimensions must align for `@` / `nn.Linear` $(M \times K) \text{ @ } (K \times N) \rightarrow (M \times N)$. Remember PyTorch images use `[Batch, Channels, Height, Width]`, while Matplotlib expects `[Height, Width, Channels]` (fix via `.permute(1, 2, 0)`).
2. **Datatype (`dtype`) Mismatch:** NumPy defaults to `float64`, whereas PyTorch neural networks default to `torch.float32` (fix via `tensor.type(torch.float32)`).
3. **Device Mismatch:** A tensor on `cpu` cannot interact with a model on `cuda:0` (fix via `.to(device)` on both data and model).

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/zzobaer09/pytorch-learning.git
cd pytorch-learning

# Install dependencies (if running locally outside Google Colab)
pip install torch torchvision torchaudio torchmetrics matplotlib pandas
```
