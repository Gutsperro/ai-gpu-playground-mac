import argparse, time, math
import torch

def sync(device: str):
    if device == "mps":
        try:
            torch.mps.synchronize()
        except Exception:
            pass

def run(device: str, size: int, repeat: int, dtype: str):
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dt = dtype_map[dtype]
    dev = torch.device(device)
    a = torch.randn(size, size, device=dev, dtype=dt)
    b = torch.randn(size, size, device=dev, dtype=dt)

    # warmup
    for _ in range(3):
        (a @ b).sum().item()
    sync(device)

    t0 = time.perf_counter()
    for _ in range(repeat):
        c = a @ b
        # prevent elimination
        c = c.sum()
        c.item()
    sync(device)
    elapsed = time.perf_counter() - t0
    flops = 2 * (size**3) * repeat  # matmul FLOPs
    tflops = flops / elapsed / 1e12
    print(f"[PyTorch] device={device} dtype={dtype} size={size} repeat={repeat} time={elapsed:.2f}s ~{tflops:.2f} TFLOP/s")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", choices=["cpu","mps"], required=True)
    p.add_argument("--size", type=int, default=8192)
    p.add_argument("--repeat", type=int, default=6)
    p.add_argument("--dtype", choices=["float32","float16","bfloat16"], default="float16")
    args = p.parse_args()
    if args.device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available. Falling back to CPU.")
        args.device = "cpu"
    run(args.device, args.size, args.repeat, args.dtype)
