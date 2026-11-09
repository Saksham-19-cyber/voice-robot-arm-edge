"""Multi-device benchmark runner."""







import os
import sys
import argparse
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.bench.benchmark import run_full_benchmark, get_hardware_info


def main():
    parser = argparse.ArgumentParser(description="Voice Robot Arm Benchmark Runner")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "gpu", "jetson"],
        help="Target device to benchmark: cpu, gpu, or jetson",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Custom output JSON path (defaults to results/bench_<device>.json)",
    )
    args = parser.parse_args()

    out_file = args.output or f"results/bench_{args.device}.json"

    if args.device == "gpu":
        import torch
        if not torch.cuda.is_available():
            print("ERROR: CUDA device not detected! Run this script on a machine with GPU (e.g. Google Colab / Kaggle).")
            sys.exit(1)
        print(f"CUDA GPU detected: {torch.cuda.get_device_name(0)}")
    elif args.device == "jetson":

        is_tegra = os.path.exists("/etc/nv_tegra_release") or "tegra" in os.uname().release.lower() if hasattr(os, "uname") else False
        if not is_tegra:
            print("Notice: Running on non-Tegra machine with --device jetson flag. Measuring edge CPU performance.")

    report = run_full_benchmark(output_path=out_file)
    report["target_device_profile"] = args.device

    with open(out_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Benchmark results successfully written to {out_file}")


if __name__ == "__main__":
    main()
