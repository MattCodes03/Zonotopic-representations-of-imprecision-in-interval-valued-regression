import threading
import torch
import psutil
import os
import gc
import time
import numpy as np
import pandas as pd


def interval_metrics(pred_low, pred_up, true_low, true_up):
    """
    Compute Dmin, Dmax, D*, MAEmid for interval predictions vs interval targets.

    Parameters
    ----------
    pred_low, pred_up : array-like, shape (N,)
        Lower/upper bounds of predicted intervals [h_i].
    true_low, true_up : array-like, shape (N,)
        Lower/upper bounds of target intervals [Y_i].

    Returns
    -------
    dict with keys 'Dmin', 'Dmax', 'Dstar', 'MAEmid'
    """
    pred_low, pred_up = np.asarray(
        pred_low, dtype=float), np.asarray(pred_up, dtype=float)
    true_low, true_up = np.asarray(
        true_low, dtype=float), np.asarray(true_up, dtype=float)

    assert np.all(
        pred_up >= pred_low), "predicted intervals must have upper >= lower"
    assert np.all(
        true_up >= true_low), "target intervals must have upper >= lower"

    gap = np.maximum(0.0, np.maximum(pred_low - true_up, true_low - pred_up))
    Dmin = gap.mean()

    dmax_per_i = np.maximum(np.abs(pred_up - true_low),
                            np.abs(true_up - pred_low))
    Dmax = dmax_per_i.mean()

    dstar_per_i = np.maximum(
        np.abs(pred_low - true_low), np.abs(pred_up - true_up))
    Dstar = dstar_per_i.mean()

    pred_mid = (pred_low + pred_up) / 2
    true_mid = (true_low + true_up) / 2
    MAEmid = np.abs(pred_mid - true_mid).mean()

    return {"Dmin": Dmin, "Dmax": Dmax, "Dstar": Dstar, "MAEmid": MAEmid}


def compare_models(results_dict, decimals=2):
    """
    results_dict: {"Model Name": {"Dmin":.., "Dmax":.., "Dstar":.., "MAEmid":..}, ...}
    Prints a Table-5-style comparison.
    """
    df = pd.DataFrame(results_dict).T[["Dmin", "Dmax", "Dstar", "MAEmid"]]
    print(df.round(decimals))
    return df


def benchmark_inference(
    predict_fn,
    X,
    name="Model",
    n_runs=20,
    warmup=5,
):
    """
    Benchmark end-to-end inference time and peak process memory.

    Measures:
        - Mean inference time
        - Standard deviation
        - Time per sample
        - Peak RSS memory increase

    Suitable for CPU-based PyTorch / NumPy models.
    """

    process = psutil.Process(os.getpid())

    gc.collect()

    for _ in range(warmup):
        _ = predict_fn(X)

    gc.collect()

    times = []

    for _ in range(n_runs):

        start = time.perf_counter()

        _ = predict_fn(X)

        end = time.perf_counter()

        times.append(end - start)

    times = np.array(times)

    mean_time = times.mean()
    std_time = times.std()

    time_per_sample_ms = (
        mean_time / len(X) * 1000
    )

    gc.collect()

    memory_samples = []
    stop_monitor = False

    def monitor_memory():

        while not stop_monitor:

            memory_samples.append(
                process.memory_info().rss
            )

            time.sleep(0.001)  # 1 ms sampling

    # Memory immediately before inference
    memory_before = process.memory_info().rss

    monitor_thread = threading.Thread(
        target=monitor_memory
    )

    monitor_thread.start()

    # Run complete inference
    _ = predict_fn(X)

    stop_monitor = True
    monitor_thread.join()

    memory_after = process.memory_info().rss

    # Include final measurement
    memory_samples.append(memory_after)

    peak_memory = max(memory_samples)

    # Memory increase relative to baseline
    peak_memory_mb = (
        max(0, peak_memory - memory_before)
        / (1024 ** 2)
    )

    results = {
        "model": name,
        "inference_time_s": mean_time,
        "inference_time_std_s": std_time,
        "time_per_sample_ms": time_per_sample_ms,
        "peak_memory_mb": peak_memory_mb
    }

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    print(
        f"Inference time : "
        f"{mean_time:.6f} ± {std_time:.6f} s"
    )

    print(
        f"Per sample     : "
        f"{time_per_sample_ms:.6f} ms"
    )

    print(
        f"Peak memory    : "
        f"{peak_memory_mb:.3f} MB"
    )

    return results
