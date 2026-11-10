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

    rep = data.get("repeated_runs", {})
    if not rep:
        print("Skipping policy_latency.png: repeated_runs missing")
        return

    fp32_vals = rep.get("fp32_p50_us", {}).get("values", [])
    int8_vals = rep.get("int8_p50_us", {}).get("values", [])
    fp32_med = rep.get("fp32_p50_us", {}).get("median", 0.0)
    int8_med = rep.get("int8_p50_us", {}).get("median", 0.0)

    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=150)

    # Plot bars for medians
    models = ["FP32 ONNX", "INT8 ONNX"]
    medians = [fp32_med, int8_med]
    colors = ["#4C72B0", "#55A868"]

    x = np.array([0, 1])
    bars = ax.bar(x, medians, width=0.45, color=colors, alpha=0.35, edgecolor="black", linewidth=1.0, label="Median p50")

    # Plot individual run points (dots)
    np.random.seed(42)
    jitter_fp32 = np.random.uniform(-0.08, 0.08, size=len(fp32_vals))
    jitter_int8 = np.random.uniform(-0.08, 0.08, size=len(int8_vals))

    ax.scatter(np.zeros(len(fp32_vals)) + jitter_fp32, fp32_vals, color="#1D3557", s=45, zorder=5, label="Individual Runs (5x)")
    ax.scatter(np.ones(len(int8_vals)) + jitter_int8, int8_vals, color="#2A9D8F", s=45, zorder=5)

    # Annotate medians
    ax.text(0, fp32_med + 3, f"Median: {fp32_med:.1f} µs\nRange: [{min(fp32_vals):.1f}, {max(fp32_vals):.1f}]", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax.text(1, int8_med + 3, f"Median: {int8_med:.1f} µs\nRange: [{min(int8_vals):.1f}, {max(int8_vals):.1f}]", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_ylabel("Single-Thread Latency p50 (µs)")
    ax.set_title("Policy Latency Distribution Across 5 Repeated Benchmark Runs", fontsize=10.5, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9.5)
    ax.set_ylim(0, max(max(fp32_vals), max(int8_vals)) + 20)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="#F8F9FA", edgecolor="#CCCCCC", fontsize=8.5)

    footnote = (
        "Footnote: Measured on Windows laptop, Intel Core i5-1235U (single thread pinned, intra_op=1, 20k iter/run).\n"
        "3.58x smaller; no measurable latency difference on this CPU; run-to-run variation exceeds the difference; cause not profiled."
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
    p_linux = "results/streamlit_memory_v4.json"
    p_win = "results/streamlit_memory_v3.json"
    if not os.path.exists(p_linux) or not os.path.exists(p_win):
        print(f"Skipping memory.png: missing {p_linux} or {p_win}")
        return

    with open(p_linux, "r", encoding="utf-8") as f:
        linux_data = json.load(f)
    with open(p_win, "r", encoding="utf-8") as f:
        win_data = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 4.4), dpi=150, gridspec_kw={'width_ratios': [3, 2]})

    # Panel 1: Linux CI (OSMesa)
    labels_l = ["Idle\nBaseline", "1 Session\n(Presets+Whisper)", "3 Concurrent\n(Presets+Whisper)"]
    vals_l = [
        linux_data.get("server_baseline_rss_mb", 54.28),
        linux_data.get("single_session", {}).get("peak_rss_mb", 88.88),
        linux_data.get("three_concurrent_sessions", {}).get("peak_rss_mb", 185.59),
    ]
    colors_l = ["#4C72B0", "#64B5CD", "#55A868"]
    bars1 = ax1.bar(labels_l, vals_l, color=colors_l, edgecolor="black", linewidth=0.8, width=0.55)
    ax1.axhline(1024, color="#C44E52", linestyle="--", linewidth=1.5, label="1,024 MB Limit")
    ax1.set_ylabel("Peak Resident Memory RSS (MB)")
    ax1.set_title("Linux CI (Headless OSMesa, Real Whisper)", fontsize=9.5, fontweight="bold")
    ax1.set_ylim(0, 1150)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", frameon=True, fontsize=7.5)

    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 25, f"{yval:.1f} MB", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Panel 2: Windows Laptop (GLFW)
    labels_w = ["Idle\nBaseline", "Active Session\n(Whisper Loaded)"]
    win_m = win_data.get("windows_laptop_measurement", {}) or win_data.get("windows_laptop_live_browser", {})
    vals_w = [
        win_m.get("idle_server_rss_mb", 127.1),
        win_m.get("tiny_en_loaded_rss_mb", win_m.get("peak_rss_mb", 205.1)),
    ]
    colors_w = ["#C44E52", "#E1974C"]
    bars2 = ax2.bar(labels_w, vals_w, color=colors_w, edgecolor="black", linewidth=0.8, width=0.45)
    ax2.set_ylabel("Memory RSS (MB)")
    ax2.set_title("Windows Laptop (GLFW Rendering)", fontsize=9.5, fontweight="bold")
    ax2.set_ylim(0, 350)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 10, f"{yval:.1f} MB", ha="center", va="bottom", fontsize=8, fontweight="bold")

    footnote = (
        "Footnote: Separated by OS. Linux measured on Ubuntu CI runner (OSMesa headless); Windows measured on Intel i5-1235U.\n"
        "Idle measures bare server; 1-session measures sequential presets + Whisper; 3-session measures concurrent load."
    )
    plt.figtext(0.5, 0.01, footnote, ha="center", fontsize=7.5, color="#444444", style="italic")
    plt.tight_layout(rect=[0, 0.07, 1, 1])

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
