import numpy as np
import os


start_index = 0

in_path = f"./results/numpy_files/mazes_long_1024/temp_75_36_topp0.92_topk2048_run0_eval_testindex_{start_index}.npy"
out_dir = "../video-diffusion/results/TATS/ema_---_---/---_-_-_-_-/samples"


data = np.load(in_path).transpose(0, 1, 4, 2, 3)
for i, s in enumerate(data):
    path = os.path.join(out_dir, f'sample_{start_index+i:04d}-0.npy')
    np.save(path, s)
