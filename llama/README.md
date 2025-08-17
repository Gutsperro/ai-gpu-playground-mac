# llama.cpp quick notes (Metal on Mac)

Build:
```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make LLAMA_METAL=1
```

Run a tiny model (you must download `.gguf` file yourself and place it under `./models`):
```bash
./main -m models/YourSmallModel.gguf -p "Hello!" -ngl 99
# Compare CPU only:
./main -m models/YourSmallModel.gguf -p "Hello!" -ngl 0
```

Key flag:
- `-ngl N` — number of GPU layers to offload. Use `-ngl 99` to offload as many as possible.
- Watch the printed **tokens per second**.
