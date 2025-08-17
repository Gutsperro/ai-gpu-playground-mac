# AI GPU Playground (Mac, Apple Silicon)

> Touch the gradient. Watch matrices fly.  
> This tiny lab lets you _feel_ why GPUs matter: measure CPU vs Metal GPU on your Mac.

![Platform](<https://img.shields.io/badge/platform-macOS%20(M1%2FM2%2FM3%2FM4)-informational>)
![PyTorch](https://img.shields.io/badge/PyTorch-MPS-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Metal-orange)
![MLX](https://img.shields.io/badge/Apple-MLX-success)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

Hands-on mini-project to _feel_ what "more GPU" means for AI training by comparing CPU vs Apple's GPU (via **Metal** / **MPS**) on your MacBook Pro (M-series).

## What you'll do

1. **PyTorch (MPS)** – measure big matrix multiply speed (the core of AI training) on `cpu` vs `mps`.
2. **TensorFlow (Metal)** – the same idea with TensorFlow & `tensorflow-metal` plugin.
3. **(Optional) llama.cpp** – run a small LLM locally and watch _tokens/sec_ jump when offloading layers to GPU.
4. **(Optional) MLX** – Apple's own framework for Apple silicon. Try quick inference / fine-tuning on small models.

> Matrix multiply is the main workload in deep learning (GEMM). Faster GEMM ⇒ faster training.

---

## 0) Prereqs

- macOS 13+ (Ventura or newer), Apple Silicon (M1/M2/M3/M4).
- Python 3.10+ recommended.
- Xcode command line tools: `xcode-select --install` (or open Xcode once).
- **Tip:** Open **Activity Monitor → Window → GPU History** to _see_ GPU usage during runs.

---

## 1) PyTorch (MPS) – CPU vs GPU benchmark

> The **MPS** backend lets PyTorch use the Apple GPU through Metal.

### Create an env & install

```bash
python3 -m venv .venv-torch
source .venv-torch/bin/activate
python -m pip install --upgrade pip
pip install torch torchvision torchaudio   # ← no CPU-only index URL
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("MPS available:", torch.backends.mps.is_available())
print("MPS built:", torch.backends.mps.is_built())
PY
```

> If `is_available()` is `True`, you’re good. If not, make sure macOS and Xcode tools are up to date.

### Run the matmul benchmark

```bash
python benchmarks/pytorch_mps_matmul.py --device cpu   --size 8192 --repeat 6 --dtype float16
python benchmarks/pytorch_mps_matmul.py --device mps   --size 8192 --repeat 6 --dtype float16
python benchmarks/pytorch_mps_matmul.py --device mps   --size 8192 --repeat 6 --dtype float32
```

You’ll see wall‑clock times and approximate TFLOP/s. The GPU (`mps`) should be significantly faster than `cpu` on larger sizes, especially with `float16`.

---

## 2) TensorFlow (Metal) – CPU vs GPU benchmark

### Create an env & install

```bash
python3 -m venv .venv-tf
source .venv-tf/bin/activate
python -m pip install --upgrade pip
pip install tensorflow tensorflow-metal
python - <<'PY'
import tensorflow as tf
print("tf:", tf.__version__)
print("GPU devices:", tf.config.list_physical_devices("GPU"))
PY
```

### Run the matmul benchmark

```bash
python benchmarks/tf_metal_matmul.py --device cpu --size 8192 --repeat 6 --dtype float16
python benchmarks/tf_metal_matmul.py --device gpu --size 8192 --repeat 6 --dtype float16
python benchmarks/tf_metal_matmul.py --device gpu --size 8192 --repeat 6 --dtype float32
```

If GPU is set up, you’ll see one Metal GPU in the device list and faster timings for `--device gpu` on larger sizes.

---

## 3) (Optional) llama.cpp – local LLM with Metal

Build llama.cpp with Metal and test tokens/sec.

```bash
# prerequisites: cmake, git, build tools (e.g. via Homebrew: brew install cmake)
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make LLAMA_METAL=1
# Download a small GGUF model (example: a 1–2B instruct model)
# (Use the model vendor’s download link; place it under ./models)
# Run with GPU offload of all layers (-ngl 99)
./main -m models/YourSmallModel.gguf -p "Say hello in Finnish." -ngl 99
```

Check the printed **tokens per second**. Compare with `-ngl 0` (CPU only). Metal offload should increase throughput.

---

## 4) (Optional) MLX (Apple’s framework)

```bash
python3 -m venv .venv-mlx
source .venv-mlx/bin/activate
pip install mlx mlx-lm
# Quick generation with a tiny MLX model
python - <<'PY'
from mlx_lm import load, generate
model, tokenizer = load("mlx-community/Qwen2.5-0.5B-Instruct-mlx")
print(generate(model, tokenizer, prompt="Miksi GPU nopeuttaa neuroverkkojen opetusta?", max_tokens=64))
PY
```

MLX is optimized for Apple silicon and can use the GPU/ANE under the hood.

---

## Why this demonstrates “many GPUs”

- **Training = tons of matrix multiplies.** GPUs do thousands of these in parallel. CPU has few wide cores; GPU has many smaller cores.
- **More (and bigger) GPUs ⇒ more throughput & memory.** You can use **data parallelism** (same model on many GPUs, split the batch) or **model/tensor parallelism** (split the model across GPUs) to scale. For huge LLMs, clusters of GPUs are linked with high‑speed interconnects.
- On a Mac you won’t train giant LLMs from scratch, but you can **feel the acceleration** and do **fine‑tuning of small models**.

---

## Repo structure

```
benchmarks/
  pytorch_mps_matmul.py
  tf_metal_matmul.py
llama/
  README.md
README.md
```

---

## Troubleshooting

- If MPS/TensorFlow GPU isn’t detected, update macOS and Xcode CLTs, and ensure you’re in the correct virtualenv.
- For PyTorch MPS, set `PYTORCH_ENABLE_MPS_FALLBACK=1` to fall back to CPU ops when an op isn’t implemented.

Enjoy!
