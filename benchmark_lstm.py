from models.CustomLSTM import CustomLSTM
import numpy as np
import os
import torch
import torch.nn as nn
import random
from models.CustomLSTM import CustomLSTM
from datasets.ecg_mit import ECG_MIT
import time
from typing import List

context_len = 64
pred_len = 64
ecg_dataset = ECG_MIT(context_len=context_len, pred_len=pred_len, data_path="/home/user/MIT-BIH.npz")

batch_size = 16
total_samples = batch_size * 20

num_iterations = 10

device = torch.device("cuda")

def batch_loader(dataset: ECG_MIT, indices: List[int], batch_size: int):
    for i in range(0, len(indices), batch_size):
        xs = []
        ys = []

        for index in indices[i : min(i + batch_size, len(indices))]:
            x, y = dataset[index]
            xs.append(x)
            ys.append(y)

        xs = torch.tensor(np.array(xs), device=device, dtype=torch.float32).unsqueeze(-1)
        yield xs, ys


indices = random.sample(range(len(ecg_dataset)), total_samples)

# ecg_dataset = ECG_MIT(context_len=context_len, pred_len=pred_len, data_path="/home/user/MIT-BIH.npz")
# model_checkpoint = "/home/user/TEMPO/lstm_64_64/checkpoint.pth"


model = CustomLSTM(context_len, pred_len)
# model.load_state_dict(torch.load(model_checkpoint), strict=False)
model = model.to(device=device)

model.eval()

total_times = []

for _ in range(num_iterations):
    start_time = time.time()
    for _, (x, _) in enumerate(batch_loader(ecg_dataset, indices, batch_size)):
        model(x)
    end_time = time.time()

    total_times.append(end_time - start_time)
    
print(f"Number of iterations: {num_iterations} | Number of batches: {total_samples/batch_size} | Batch Size: {batch_size}")
print(f"Average time taken to run : {np.average(total_times)}")
print(f"Std. of time taken to run : {np.std(total_times)}")


if not os.path.exists("logs"):
    os.mkdir("logs")

with open(os.path.join("logs", f"LSTM_benchmark.txt"), "w") as f:
    f.write(f"Number of iterations: {num_iterations} | Number of batches: {total_samples/batch_size} | Batch Size: {batch_size} \n")
    f.write(f"Average time taken to run : {np.average(total_times)} \n")
    f.write(f"Std. of time taken to run : {np.std(total_times)}")