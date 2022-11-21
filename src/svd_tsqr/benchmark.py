
import time
import cupy as cp
import numpy as np


from cupyx.profiler import benchmark


def svd(A, mod):
    U, S, V = mod.linalg.svd(B)
    return U, S, V


#%%
N = 10000

B = cp.random.random((N, N), dtype=cp.float32)

start_gpu = cp.cuda.Event()
end_gpu = cp.cuda.Event()

start_gpu.record()
start_cpu = time.perf_counter()
out = svd(B, cp)
end_cpu = time.perf_counter()
end_gpu.record()
end_gpu.synchronize()
t_gpu = cp.cuda.get_elapsed_time(start_gpu, end_gpu) / 1000.0

print(f"GPU Elapsed Time: {t_gpu} secs")
