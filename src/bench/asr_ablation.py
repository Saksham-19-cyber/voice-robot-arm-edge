"""
ASR Speed Ablation Study:
Tests configuration changes one at a time on faster-whisper tiny.en:
1. Baseline: default settings (beam_size=5, vad_filter=True, language=None, condition_on_prev=True)
2. beam_size=1
3. vad_filter=False (for clips < 3s)
4. cpu_threads=4 (vs default 1 or system default)
5. language="en" (fixed language bypasses language detection)
6. condition_on_previous_text=False
7. Optimal combined combination (keeps all changes that lower latency without altering transcript)
"""

import os
import sys
import time
import json
import numpy as np
from faster_whisper import WhisperModel

def run_asr_ablation():
    audio_path = "results/sample_speech.wav"
    assert os.path.exists(audio_path), f"Audio missing: {audio_path}"
    
    # Baseline model
    print("Testing ASR configurations one by one...", flush=True)
    results = []
    
    experiments = [
        ("Baseline (defaults)", dict(model_threads=4, beam_size=5, vad_filter=True, language=None, condition_on_prev=True)),
        ("+ beam_size=1", dict(model_threads=4, beam_size=1, vad_filter=True, language=None, condition_on_prev=True)),
        ("+ vad_filter=False (<3s)", dict(model_threads=4, beam_size=5, vad_filter=False, language=None, condition_on_prev=True)),
        ("+ cpu_threads=1", dict(model_threads=1, beam_size=5, vad_filter=True, language=None, condition_on_prev=True)),
        ("+ cpu_threads=8", dict(model_threads=8, beam_size=5, vad_filter=True, language=None, condition_on_prev=True)),
        ("+ language='en' (fixed)", dict(model_threads=4, beam_size=5, vad_filter=True, language="en", condition_on_prev=True)),
        ("+ condition_on_prev=False", dict(model_threads=4, beam_size=5, vad_filter=True, language=None, condition_on_prev=False)),
        ("Optimal Combined", dict(model_threads=4, beam_size=1, vad_filter=False, language="en", condition_on_prev=False)),
    ]
    
    # Cache models by thread count to avoid reloading
    models = {}
    for threads in [1, 4, 8]:
        models[threads] = WhisperModel("tiny.en", device="cpu", compute_type="int8", cpu_threads=threads)
        
    for name, cfg in experiments:
        m = models[cfg["model_threads"]]
        
        # Warmup pass
        _ = list(m.transcribe(
            audio_path,
            beam_size=cfg["beam_size"],
            vad_filter=cfg["vad_filter"],
            language=cfg["language"],
            condition_on_previous_text=cfg["condition_on_prev"],
        )[0])
        
        # 5 timed runs
        latencies = []
        transcript = ""
        for _ in range(5):
            t0 = time.perf_counter()
            segs, _ = m.transcribe(
                audio_path,
                beam_size=cfg["beam_size"],
                vad_filter=cfg["vad_filter"],
                language=cfg["language"],
                condition_on_previous_text=cfg["condition_on_prev"],
            )
            text = " ".join([s.text.strip() for s in segs]).strip()
            latencies.append((time.perf_counter() - t0) * 1000)
            transcript = text
            
        mean_lat = float(np.mean(latencies))
        std_lat = float(np.std(latencies))
        p50_lat = float(np.median(latencies))
        
        row = {
            "config_name": name,
            "mean_latency_ms": round(mean_lat, 1),
            "std_latency_ms": round(std_lat, 1),
            "p50_latency_ms": round(p50_lat, 1),
            "transcript": transcript,
            "transcript_preserved": (transcript.lower() == "reach to the left."),
            "speedup_vs_baseline": None,
        }
        results.append(row)
        print(f"[{name}] {mean_lat:.1f} +/- {std_lat:.1f} ms | transcript: '{transcript}'", flush=True)
        
    base_lat = results[0]["mean_latency_ms"]
    for r in results:
        r["speedup_vs_baseline"] = round(base_lat / r["mean_latency_ms"], 2)
        
    with open("results/asr_ablation.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved ASR ablation results to results/asr_ablation.json", flush=True)
    return results

if __name__ == "__main__":
    run_asr_ablation()
