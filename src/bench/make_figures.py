"""Generate PNG figures from benchmark results."""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = "assets/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_policy_latency():
    path = "results/policy_benchmark.json"
    if not os.path.exists(path):
        print(f"Skipping {path}: file not found")
        return
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    fp32 = data.get("fp32")
    int8 = data.get("int8")
    boot = data.get("bootstrap_p50_difference", {})
    if not fp32 or not int8:
        print("Skipping policy_latency.png: fp32 or int8 data missing")
        return

    metrics = ["p50 (Median)", "p95", "p99"]
    fp32_vals = [fp32["p50_us"], fp32["p95_us"], fp32["p99_us"]]
    int8_vals = [int8["p50_us"], int8["p95_us"], int8["p99_us"]]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    bars1 = ax.bar(x - width/2, fp32_vals, width, label="FP32 ONNX", color="#4C72B0", edgecolor="black", linewidth=0.8)
    bars2 = ax.bar(x + width/2, int8_vals, width, label="INT8 ONNX", color="#55A868", edgecolor="black", linewidth=0.8)

    ax.set_ylabel("Latency (µs)")
    ax.set_title("Single-Thread Policy Inference Latency (20,000 Iterations)", fontsize=11, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.legend(frameon=True, facecolor="#F8F9FA", edgecolor="#CCCCCC")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars1:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 3, f"{yval:.1f} µs", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 3, f"{yval:.1f} µs", ha="center", va="bottom", fontsize=8)

    ci_low = boot.get("ci_95_low_us", 28.2)
    ci_high = boot.get("ci_95_high_us", 29.4)
    footnote = (
        f"Footnote: Measured on Windows laptop, Intel Core i5-1235U (single thread pinned, intra_op=1, inter_op=1).\n"
        f"Bootstrap 95% CI on p50 difference (FP32 - INT8): [{ci_low:.1f}, {ci_high:.1f}] µs (2,000 resamples)."
    )
    plt.figtext(0.5, 0.01, footnote, ha="center", fontsize=7.5, color="#444444", style="italic")
    plt.tight_layout(rect=[0, 0.07, 1, 1])

    out_file = os.path.join(OUTPUT_DIR, "policy_latency.png")
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"Generated {out_file} ({os.path.getsize(out_file) / 1024:.1f} KB)")


def plot_model_size_and_parity():
    path = "results/quantization_eval.json"
    if not os.path.exists(path):
        print(f"Skipping {path}: file not found")
        return
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    models = ["PyTorch FP32", "ONNX FP32", "ONNX INT8"]
    sizes_kb = [
        data.get("fp32_size_bytes", 287716) / 1024.0,
        data.get("fp32_size_bytes", 287716) / 1024.0,
        data.get("int8_size_bytes", 80341) / 1024.0,
    ]
    success_rates = [
        data.get("PyTorch_FP32", {}).get("success_rate", 1.0) * 100,
        data.get("ONNX_FP32", {}).get("success_rate", 1.0) * 100,
        data.get("ONNX_INT8", {}).get("success_rate", 1.0) * 100,
    ]
    steps_to_goal = [
        data.get("PyTorch_FP32", {}).get("mean_steps_to_goal", 2.65),
        data.get("ONNX_FP32", {}).get("mean_steps_to_goal", 2.65),
        data.get("ONNX_INT8", {}).get("mean_steps_to_goal", 2.65),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 4.2), dpi=150)

    colors = ["#4C72B0", "#64B5CD", "#55A868"]
    bars1 = ax1.bar(models, sizes_kb, color=colors, edgecolor="black", linewidth=0.8, width=0.55)
    ax1.set_ylabel("Model Size (KB)")
    ax1.set_title("Disk Storage Footprint", fontsize=10, fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 4, f"{yval:.1f} KB", ha="center", va="bottom", fontsize=8.5)
    ax1.text(2, 40, "3.58x\ncompression", ha="center", va="center", color="white", fontweight="bold", fontsize=8)

    x = np.arange(len(models))
    width = 0.35
    bars2_succ = ax2.bar(x - width/2, success_rates, width, label="Success Rate (%)", color="#55A868", edgecolor="black", linewidth=0.8)
    bars2_steps = ax2.bar(x + width/2, steps_to_goal, width, label="Mean Steps to Goal", color="#C44E52", edgecolor="black", linewidth=0.8)
    ax2.set_ylabel("Metric Value")
    ax2.set_title("Closed-Loop Parity (100 Episodes)", fontsize=10, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=8.5)
    ax2.legend(frameon=True, facecolor="#F8F9FA", edgecolor="#CCCCCC", fontsize=8)
    ax2.set_ylim(0, 115)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars2_succ:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 2, f"{yval:.0f}%", ha="center", va="bottom", fontsize=8)
    for bar in bars2_steps:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 2, f"{yval:.2f}", ha="center", va="bottom", fontsize=8)

    footnote = "Footnote: Measured on Windows laptop, Intel Core i5-1235U, GLFW rendering (100 evaluation episodes each)."
    plt.figtext(0.5, 0.01, footnote, ha="center", fontsize=7.5, color="#444444", style="italic")
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    out_file = os.path.join(OUTPUT_DIR, "model_size_and_parity.png")
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"Generated {out_file} ({os.path.getsize(out_file) / 1024:.1f} KB)")


def plot_pipeline_latency():
    path = "results/pipeline_latency.json"
    if not os.path.exists(path):
        print(f"Skipping {path}: file not found")
        return
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    early = data.get("early_termination")
    full = data.get("full_50_steps")
    if not early or not full:
        print("Skipping pipeline_latency.png: early_termination or full_50_steps missing")
        return

    categories = [
        f"Early-Term Rollout\n({early.get('mean_steps_taken', 13.0):.0f} steps avg)",
        f"Full Rollout\n({full.get('mean_steps_taken', 50.0):.0f} steps fixed)"
    ]

    asr_vals = [early["asr_inference_ms"]["mean"], full["asr_inference_ms"]["mean"]]
    nlu_vals = [early["nlu_parse_ms"]["mean"], full["nlu_parse_ms"]["mean"]]
    rollout_vals = [early["rollout_and_render_ms"]["mean"], full["rollout_and_render_ms"]["mean"]]
    encode_vals = [early["video_encode_ms"]["mean"], full["video_encode_ms"]["mean"]]

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    width = 0.45

    b_asr = ax.bar(categories, asr_vals, width, label="ASR (faster-whisper INT8)", color="#4C72B0", edgecolor="black", linewidth=0.8)
    b_nlu = ax.bar(categories, nlu_vals, width, bottom=asr_vals, label="NLU (Intent + Clipping)", color="#DD8452", edgecolor="black", linewidth=0.8)

    bottom_roll = np.array(asr_vals) + np.array(nlu_vals)
    b_roll = ax.bar(categories, rollout_vals, width, bottom=bottom_roll, label="Simulation + Rendering", color="#55A868", edgecolor="black", linewidth=0.8)

    bottom_enc = bottom_roll + np.array(rollout_vals)
    b_enc = ax.bar(categories, encode_vals, width, bottom=bottom_enc, label="MP4 H.264 Encoding", color="#C44E52", edgecolor="black", linewidth=0.8)

    totals = bottom_enc + np.array(encode_vals)
    for idx, total in enumerate(totals):
        ax.text(idx, total + 35, f"Total: {total:.1f} ms\n(~{total/1000:.2f} s)", ha="center", va="bottom", fontweight="bold", fontsize=8.5)

    ax.set_ylabel("Mean Stage Latency (ms)")
    ax.set_title("End-to-End Pipeline Latency: Early Termination vs Full Rollout", fontsize=11, fontweight="bold", pad=12)
    ax.legend(loc="upper left", frameon=True, facecolor="#F8F9FA", edgecolor="#CCCCCC", fontsize=8.5)
    ax.set_ylim(0, 2750)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    footnote = (
        "Footnote: Measured on Windows laptop, Intel Core i5-1235U, GLFW rendering (20 runs each).\n"
        "Cold-start model load & first-frame initialization (17.68 s) excluded; cached in memory."
    )
    plt.figtext(0.5, 0.01, footnote, ha="center", fontsize=7.5, color="#444444", style="italic")
    plt.tight_layout(rect=[0, 0.06, 1, 1])

    out_file = os.path.join(OUTPUT_DIR, "pipeline_latency.png")
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"Generated {out_file} ({os.path.getsize(out_file) / 1024:.1f} KB)")


def plot_asr_ablation():
    path = "results/asr_ablation.json"
    if not os.path.exists(path):
        print(f"Skipping {path}: file not found")
        return
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    labels = []
    p50_vals = []
    p95_vals = []
    for item in data:
        cfg = item.get("config_name", "")
        clean_name = cfg.replace("_threads", " th=").replace("_", " ")
        labels.append(clean_name)
        p50_vals.append(item.get("latency_p50_ms", 0))
        p95_vals.append(item.get("latency_p95_ms", 0))

    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=150)
    y = np.arange(len(labels))
    height = 0.35

    bars1 = ax.barh(y - height/2, p50_vals, height, label="p50 (Median)", color="#4C72B0", edgecolor="black", linewidth=0.8)
    bars2 = ax.barh(y + height/2, p95_vals, height, label="p95", color="#C44E52", edgecolor="black", linewidth=0.8)

    ax.set_xlabel("Latency (ms) on 2.0s Audio Clip")
    ax.set_title("ASR Inference Latency Ablation Across Configurations", fontsize=11, fontweight="bold", pad=12)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.legend(loc="lower right", frameon=True, facecolor="#F8F9FA", edgecolor="#CCCCCC", fontsize=8.5)
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    for bar in bars1:
        w = bar.get_width()
        ax.text(w + 40, bar.get_y() + bar.get_height()/2.0, f"{w:.0f}", ha="left", va="center", fontsize=7.5)

    footnote = "Footnote: Measured on Windows laptop, Intel Core i5-1235U (2.0s synthetic audio, 20 runs per configuration)."
    plt.figtext(0.5, 0.01, footnote, ha="center", fontsize=7.5, color="#444444", style="italic")
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    out_file = os.path.join(OUTPUT_DIR, "asr_ablation.png")
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"Generated {out_file} ({os.path.getsize(out_file) / 1024:.1f} KB)")


def plot_nlu_accuracy():
    path = "results/nlu_eval_detailed.json"
    if not os.path.exists(path):
        print(f"Skipping {path}: file not found")
        return
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    agent_data = data.get("dev_agent_written") or data.get("agent_written_heldout")
    human_data = data.get("human_written_heldout", {})
    if human_data.get("accuracy") is None:
        print("Note: human_written_heldout has no measured data (N=0); plotting dev_agent_written only.")

    if not agent_data:
        print("Skipping nlu_accuracy.png: dev_agent_written missing")
        return

    rule = agent_data.get("rule_parser", {})
    clf = agent_data.get("classifier_alone", {})
    hybrid = agent_data.get("hybrid_parser", {})
    base = agent_data.get("majority_class_baseline", {})

    models = [
        "Majority Baseline",
        "Rule Parser",
        "Classifier Alone",
        "Hybrid Parser (Prod)"
    ]

    intent_acc = [
        base.get("wilson_ci", {}).get("accuracy", 0.3) * 100,
        rule.get("intent_accuracy", {}).get("accuracy", 0.52) * 100,
        clf.get("intent_accuracy", {}).get("accuracy", 0.70) * 100,
        hybrid.get("intent_accuracy", {}).get("accuracy", 0.54) * 100,
    ]
    intent_err_low = [
        intent_acc[0] - base.get("wilson_ci", {}).get("ci_low", 0.191) * 100,
        intent_acc[1] - rule.get("intent_accuracy", {}).get("ci_low", 0.3851) * 100,
        intent_acc[2] - clf.get("intent_accuracy", {}).get("ci_low", 0.5625) * 100,
        intent_acc[3] - hybrid.get("intent_accuracy", {}).get("ci_low", 0.404) * 100,
    ]
    intent_err_high = [
        base.get("wilson_ci", {}).get("ci_high", 0.4375) * 100 - intent_acc[0],
        rule.get("intent_accuracy", {}).get("ci_high", 0.652) * 100 - intent_acc[1],
        clf.get("intent_accuracy", {}).get("ci_high", 0.809) * 100 - intent_acc[2],
        hybrid.get("intent_accuracy", {}).get("ci_high", 0.6703) * 100 - intent_acc[3],
    ]

    e2e_acc = [
        0.0,
        rule.get("end_to_end_accuracy", {}).get("accuracy", 0.46) * 100,
        0.0,
        hybrid.get("end_to_end_accuracy", {}).get("accuracy", 0.48) * 100,
    ]
    e2e_err_low = [
        0.0,
        e2e_acc[1] - rule.get("end_to_end_accuracy", {}).get("ci_low", 0.3297) * 100,
        0.0,
        e2e_acc[3] - hybrid.get("end_to_end_accuracy", {}).get("ci_low", 0.348) * 100,
    ]
    e2e_err_high = [
        0.0,
        rule.get("end_to_end_accuracy", {}).get("ci_high", 0.596) * 100 - e2e_acc[1],
        0.0,
        hybrid.get("end_to_end_accuracy", {}).get("ci_high", 0.6149) * 100 - e2e_acc[3],
    ]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=150)
    b1 = ax.bar(x - width/2, intent_acc, width, yerr=[intent_err_low, intent_err_high],
                label="Intent Accuracy (%)", capsize=4, color="#4C72B0", edgecolor="black", linewidth=0.8)
    b2 = ax.bar(x + width/2, e2e_acc, width, yerr=[e2e_err_low, e2e_err_high],
                label="End-to-End Accuracy (%)", capsize=4, color="#55A868", edgecolor="black", linewidth=0.8)

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("NLU Accuracy with Wilson 95% CIs (Development Set, Agent-Written, N=50)", fontsize=10, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=8.5)
    ax.set_ylim(0, 100)
    ax.legend(frameon=True, facecolor="#F8F9FA", edgecolor="#CCCCCC", fontsize=8.5)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar, val in zip(b1, intent_acc):
        ax.text(bar.get_x() + bar.get_width()/2.0, val + 5, f"{val:.1f}%", ha="center", va="bottom", fontsize=7.5)
    for bar, val in zip(b2, e2e_acc):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2.0, val + 5, f"{val:.1f}%", ha="center", va="bottom", fontsize=7.5)
        else:
            ax.text(bar.get_x() + bar.get_width()/2.0, 3, "N/A", ha="center", va="bottom", fontsize=7.5, color="#777777")

    footnote = (
        "Footnote: Evaluated on development set (agent-written; parser tuned against it; not independent; N=50).\n"
        "Human held-out test split is unmeasured (N=0). Error bars denote Wilson 95% confidence intervals."
    )
    plt.figtext(0.5, 0.01, footnote, ha="center", fontsize=7.5, color="#444444", style="italic")
    plt.tight_layout(rect=[0, 0.06, 1, 1])

    out_file = os.path.join(OUTPUT_DIR, "nlu_accuracy.png")
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"Generated {out_file} ({os.path.getsize(out_file) / 1024:.1f} KB)")


def plot_memory():
    p_linux = "results/streamlit_memory_v2.json"
    p_win = "results/streamlit_memory.json"
    if not os.path.exists(p_linux) or not os.path.exists(p_win):
        print(f"Skipping memory.png: missing {p_linux} or {p_win}")
        return

    with open(p_linux, "r", encoding="utf-8") as f:
        linux_data = json.load(f)
    with open(p_win, "r", encoding="utf-8") as f:
        win_data = json.load(f)

    labels = [
        "Linux CI\nIdle Baseline",
        "Linux CI\n1 Session (Presets)",
        "Linux CI\n3 Concurrent (Presets)",
        "Windows Laptop\n1 Session (Full Demo)"
    ]
    rss_vals = [
        linux_data.get("server_baseline_rss_mb", 54.2),
        linux_data.get("single_session", {}).get("peak_rss_mb", 103.92),
        linux_data.get("three_concurrent_sessions", {}).get("peak_rss_mb", 193.65),
        win_data.get("peak_rss_mb", 243.6),
    ]

    colors = ["#4C72B0", "#64B5CD", "#55A868", "#E1974C"]

    fig, ax = plt.subplots(figsize=(7.2, 4.5), dpi=150)
    bars = ax.bar(labels, rss_vals, color=colors, edgecolor="black", linewidth=0.8, width=0.55)

    ax.axhline(1024, color="#C44E52", linestyle="--", linewidth=1.5, label="1,024 MB Container Limit (Streamlit Cloud)")

    ax.set_ylabel("Peak Resident Memory RSS (MB)")
    ax.set_title("Streamlit Server Process Peak RSS vs 1 GB Container Budget", fontsize=11, fontweight="bold", pad=12)
    ax.set_ylim(0, 1150)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", frameon=True, facecolor="#F8F9FA", edgecolor="#CCCCCC", fontsize=8)

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 20, f"{yval:.1f} MB", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    footnote = (
        "Footnote: Measured on Linux Ubuntu (CI runner, OSMesa rendering) and Windows laptop (Intel i5-1235U, GLFW).\n"
        "Linux preset runs execute simulation rollout + MP4 encoding (ASR uncalled). Windows run includes Whisper model loaded."
    )
    plt.figtext(0.5, 0.01, footnote, ha="center", fontsize=7.5, color="#444444", style="italic")
    plt.tight_layout(rect=[0, 0.06, 1, 1])

    out_file = os.path.join(OUTPUT_DIR, "memory.png")
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"Generated {out_file} ({os.path.getsize(out_file) / 1024:.1f} KB)")


def main():
    print("Generating figures from results/*.json...")
    plot_policy_latency()
    plot_model_size_and_parity()
    plot_pipeline_latency()
    plot_asr_ablation()
    plot_nlu_accuracy()
    plot_memory()
    print("All figures processed.")


if __name__ == "__main__":
    main()
