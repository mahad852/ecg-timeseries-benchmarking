import numpy as np
from datasets.ecg_mit import ECG_MIT
import random
import os
import time

import torch
from chronos import ChronosPipeline

context_len = 64
pred_len = 64
ecg_dataset = ECG_MIT(context_len=context_len, pred_len=pred_len, data_path="/home/user/MIT-BIH.npz")

batch_size = 16
total_samples = batch_size * 20

num_iterations = 10

def single_loader(dataset: ECG_MIT, indices: list[int]):
    for index in indices:
        x, y = dataset[index]
        yield [x], [y]

def batch_loader(dataset: ECG_MIT, indices: list[int], batch_size: int):
    for i in range(0, len(indices), batch_size):
        xs = []
        ys = []

        for index in indices[i : min(i + batch_size, len(indices))]:
            x, y = dataset[index]
            xs.append(x)
            ys.append(y)

        yield xs, ys


indices = random.sample(range(len(ecg_dataset)), total_samples)

model_type = "tiny"

pipeline = ChronosPipeline.from_pretrained(
    f"amazon/chronos-t5-{model_type}",
    device_map="cuda:0",  # use "cpu" for CPU inference and "mps" for Apple Silicon
    torch_dtype=torch.bfloat16,
)
# pipeline.tokenizer.centers = pipeline.tokenizer.centers.to(torch.device("cuda:0"))
# pipeline.tokenizer.boundaries = pipeline.tokenizer.boundaries.to(torch.device("cuda:0"))
# print(pipeline.model.device, pipeline.tokenizer.centers.device)


total_times = []

for i in range(num_iterations):
    start_time = time.time()
    for _, (x, _) in enumerate(batch_loader(ecg_dataset, indices, batch_size)):
        x = torch.tensor(np.array(x))
        pipeline.predict(
            context=x,
            prediction_length=pred_len,
            num_samples=20,
        )

    end_time = time.time()

    print(f"Iteration: {i} Time taken: {end_time - start_time}")

    total_times.append(end_time - start_time)

print(f"Number of iterations: {num_iterations} | Number of batches: {total_samples/batch_size} | Batch Size: {batch_size}")
print(f"Average time taken to run : {np.average(total_times)}")
print(f"Std. of time taken to run : {np.std(total_times)}")


if not os.path.exists("logs"):
    os.mkdir("logs")

with open(os.path.join("logs", f"Chronos_benchmark_{model_type}.txt"), "w") as f:
    f.write(f"Number of iterations: {num_iterations} | Number of batches: {total_samples/batch_size} | Batch Size: {batch_size} \n")
    f.write(f"Average time taken to run : {np.average(total_times)} \n")
    f.write(f"Std. of time taken to run : {np.std(total_times)}")