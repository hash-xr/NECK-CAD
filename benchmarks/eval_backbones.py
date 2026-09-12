import time
import logging
import torch
from typing import List, Dict, Any
import numpy as np

from models.backbone_factory import BackboneFactory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NECK-CAD.Benchmark")


def benchmark_backbones(
    models_to_test: List[str] | None = None,
    batch_size: int = 1,
    num_warmup: int = 3,
    num_runs: int = 10,
    image_size: int = 224,
) -> List[Dict[str, Any]]:
    """
    Evaluates vision backbones on latency, memory allocation, and vector dimensions.

    Args:
        models_to_test (List[str]): List of backbone keys to evaluate.
        batch_size (int): Number of image patches per batch.
        num_warmup (int): Warmup forward passes before timing.
        num_runs (int): Number of timed iterations.
        image_size (int): Height and width of target input tensor.

    Returns:
        List[Dict[str, Any]]: Benchmark results table.
    """
    if models_to_test is None:
        models_to_test = ["mock", "resnet50"]  # Add 'phikon', 'uni', 'virchow' when weights are downloaded

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running backbone evaluation suite on device: {device}")

    results = []

    for model_name in models_to_test:
        logger.info(f"\n--- Benchmarking Backbone: '{model_name}' ---")
        try:
            # Instantiate model
            model = BackboneFactory.create(model_name=model_name, pretrained=True).to(device)
            model.eval()

            # Generate synthetic patch tensor
            dummy_input = torch.randn(batch_size, 3, image_size, image_size, device=device)

            # Warmup runs
            with torch.no_grad():
                for _ in range(num_warmup):
                    _ = model(dummy_input)

            # Timed latency runs
            latencies = []
            output: torch.Tensor | None = None

            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                for _ in range(num_runs):
                    start_event.record()
                    with torch.no_grad():
                        output = model(dummy_input)
                    end_event.record()
                    torch.cuda.synchronize()
                    latencies.append(start_event.elapsed_time(end_event))  # ms
                
                peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            else:
                for _ in range(num_runs):
                    t0 = time.perf_counter()
                    with torch.no_grad():
                        output = model(dummy_input)
                    t1 = time.perf_counter()
                    latencies.append((t1 - t0) * 1000.0)  # ms
                peak_memory_mb = 0.0  # CPU RAM tracking handled separately

            mean_latency = float(np.mean(latencies))
            std_latency = float(np.std(latencies))

            if output is not None:
                embedding_dim = output.shape[1]
            else:
                raise ValueError("No timed runs were executed, cannot determine embedding dimension.")

            res = {
                "Model Key": model_name,
                "Embedding Dim": embedding_dim,
                "Mean Latency (ms)": round(mean_latency, 2),
                "Std Latency (ms)": round(std_latency, 2),
                "Peak VRAM (MB)": round(peak_memory_mb, 2),
                "Device": str(device),
            }
            results.append(res)
            logger.info(f"Success: {res}")

        except Exception as e:
            logger.error(f"Failed to benchmark model '{model_name}': {e}")
            results.append({"Model Key": model_name, "Error": str(e)})

    return results


def print_summary_table(results: List[Dict[str, Any]]):
    """Prints a formatted summary table to stdout."""
    print("\n" + "=" * 80)
    print(f"{'MODEL BENCHMARK SUMMARY':^80}")
    print("=" * 80)
    header = f"{'Model':<12} | {'Dim':<6} | {'Mean (ms)':<10} | {'Std (ms)':<9} | {'VRAM (MB)':<10} | {'Device':<8}"
    print(header)
    print("-" * 80)
    for r in results:
        if "Error" in r:
            print(f"{r['Model Key']:<12} | ERROR: {r['Error']}")
        else:
            print(
                f"{r['Model Key']:<12} | {r['Embedding Dim']:<6} | "
                f"{r['Mean Latency (ms)']:<10} | {r['Std Latency (ms)']:<9} | "
                f"{r['Peak VRAM (MB)']:<10} | {r['Device']:<8}"
            )
    print("=" * 80 + "\n")


if __name__ == "__main__":
    benchmark_results = benchmark_backbones()
    print_summary_table(benchmark_results)