import numpy as np
from datasets.ecg_mit import ECG_MIT
import random
import os
import time

from lag_llama.gluon.estimator import LagLlamaEstimator
import torch
from huggingface_hub import snapshot_download
from gluonts.evaluation import make_evaluation_predictions, Evaluator

from gluonts.dataset.repository.datasets import get_dataset

from gluonts.dataset.pandas import PandasDataset
import pandas as pd

if not os.path.exists("lagllama_weights"):
    snapshot_download("time-series-foundation-models/Lag-Llama", local_dir="lagllama_weights")

context_len = 64
pred_len = 64
ecg_dataset = ECG_MIT(context_len=context_len, pred_len=pred_len, data_path="/home/user/MIT-BIH.npz")

batch_size = 16
total_samples = batch_size * 20

num_iterations = 10


indices = random.sample(range(len(ecg_dataset)), total_samples)

device = torch.device("cuda:0") # You can switch this to CPU or other GPUs if you'd like, depending on your environment

ckpt = torch.load("lagllama_weights/lag-llama.ckpt", map_location=device) # Uses GPU since in this Colab we use a GPU.
estimator_args = ckpt["hyper_parameters"]["model_kwargs"]

rope_scaling_arguments = {
    "type": "linear",
    "factor": max(1.0, (context_len + pred_len) / estimator_args["context_length"]),
}

estimator = LagLlamaEstimator(
    ckpt_path="lagllama_weights/lag-llama.ckpt",
    prediction_length=pred_len,
    context_length=context_len, # Lag-Llama was trained with a context length of 32, but can work with any context length

    # estimator args
    input_size=estimator_args["input_size"],
    n_layer=estimator_args["n_layer"],
    n_embd_per_head=estimator_args["n_embd_per_head"],
    n_head=estimator_args["n_head"],
    scaling=estimator_args["scaling"],
    time_feat=estimator_args["time_feat"],
    rope_scaling=None,
    batch_size=batch_size,
    num_parallel_samples=100,
    device=device,
)

lightning_module = estimator.create_lightning_module()
transformation = estimator.create_transformation()
predictor = estimator.create_predictor(transformation, lightning_module)


def get_lag_llama_predictions(dataset, num_samples=100):
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=dataset,
        predictor=predictor,
        num_samples=num_samples
    )

    forecasts = list(forecast_it)
    tss = list(ts_it)

    return forecasts, tss


total_times = []

def build_timeseries(dataset: ECG_MIT, indices: list[int]):
    targets = None

    item_ids = []

    for index in indices:
        x, y = dataset[index]
        vals = np.append(x, y)

        if targets is None:
            targets = vals
        else:
            targets = np.append(targets, vals)

        item_ids.extend([str(index)] * (x.shape[0] + y.shape[0]))

    df = pd.DataFrame(data={"item_id" : item_ids, "target" : targets})
    df["item_id"] = df["item_id"].astype("string")
    df["target"] = df["target"].astype("float32")
    df = df.set_index(pd.date_range('1990', freq='3ms', periods=df.shape[0]))

    dataset = PandasDataset.from_long_dataframe(df, target="target", item_id="item_id")
    return dataset
    

dataset = build_timeseries(ecg_dataset, indices)

backtest_dataset = dataset
num_samples = 20 # number of samples sampled from the probability distribution for each timestep


for i in range(num_iterations):
    start_time = time.time()
    get_lag_llama_predictions(backtest_dataset, num_samples=num_samples)
    end_time = time.time()

    print("Iteration:", i, "time taken:", end_time - start_time)

    total_times.append(end_time - start_time)

print(f"Number of iterations: {num_iterations} | Number of batches: {total_samples/batch_size} | Batch Size: {batch_size}")
print(f"Average time taken to run : {np.average(total_times)}")
print(f"Std. of time taken to run : {np.std(total_times)}")


if not os.path.exists("logs"):
    os.mkdir("logs")

with open(os.path.join("logs", f"LagLlama_benchmark.txt"), "w") as f:
    f.write(f"Number of iterations: {num_iterations} | Number of batches: {total_samples/batch_size} | Batch Size: {batch_size} \n")
    f.write(f"Average time taken to run : {np.average(total_times)} \n")
    f.write(f"Std. of time taken to run : {np.std(total_times)}")