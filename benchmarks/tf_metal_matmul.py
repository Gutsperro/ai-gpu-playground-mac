import argparse, time, math, os
import tensorflow as tf

def set_device(device: str):
    if device == "cpu":
        tf.config.set_visible_devices([], "GPU")  # hide GPU
    else:
        # ensure GPU is visible
        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            print("No GPU found; running on CPU")
        else:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass

def run(device: str, size: int, repeat: int, dtype: str):
    dt_map = {"float32": tf.float32, "float16": tf.float16, "bfloat16": tf.bfloat16}
    dt = dt_map[dtype]
    with tf.device("/CPU:0" if device=="cpu" else "/GPU:0"):
        a = tf.random.normal((size,size), dtype=dt)
        b = tf.random.normal((size,size), dtype=dt)
        # warmup
        for _ in range(3):
            c = tf.matmul(a,b)
            _ = tf.reduce_sum(c).numpy()

        t0 = time.perf_counter()
        for _ in range(repeat):
            c = tf.matmul(a,b)
            _ = tf.reduce_sum(c).numpy()
        elapsed = time.perf_counter() - t0
    flops = 2 * (size**3) * repeat
    tflops = flops / elapsed / 1e12
    print(f"[TensorFlow] device={device} dtype={dtype} size={size} repeat={repeat} time={elapsed:.2f}s ~{tflops:.2f} TFLOP/s")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", choices=["cpu","gpu"], required=True)
    p.add_argument("--size", type=int, default=8192)
    p.add_argument("--repeat", type=int, default=6)
    p.add_argument("--dtype", choices=["float32","float16","bfloat16"], default="float16")
    args = p.parse_args()
    set_device(args.device)
    run(args.device, args.size, args.repeat, args.dtype)
