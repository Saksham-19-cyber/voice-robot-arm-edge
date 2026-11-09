"""
Unit tests for ASR speech transcription.
Verifies faster-whisper tiny.en transcriber on:
1. Real speech audio (results/sample_speech.wav)
2. Synthetic pure silence (all zeros)
3. Faint Gaussian white noise below energy threshold
4. Energy RMS calculations and hallucination suppression
"""

import os
import pytest
import numpy as np
import wave

from src.asr.transcriber import WhisperASR, analyze_audio_source


@pytest.fixture(scope="module")
def transcriber():
    return WhisperASR(model_size="tiny.en", compute_type="int8", cpu_threads=1)


def test_transcribe_sample_wav(transcriber):
    sample_path = "results/sample_speech.wav"
    assert os.path.exists(sample_path), f"Audio fixture missing: {sample_path}"

    result = transcriber.transcribe(sample_path)
    assert result["text"] != ""
    assert result["latency_sec"] > 0
    # The speech in sample_speech.wav was "Reach to the left."
    text_lower = result["text"].lower()
    assert "reach" in text_lower or "left" in text_lower
    assert result["is_silent"] is False
    assert result["energy_rms"] > 0.01


def test_transcribe_silent_audio(transcriber, tmp_path):
    # Generate 1 second of pure silence
    silence_file = str(tmp_path / "silence.wav")
    sr = 16000
    with wave.open(silence_file, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(b"\x00" * (sr * 2))

    result = transcriber.transcribe(silence_file)
    assert result["audio_duration_sec"] >= 0.9
    assert result["text"].strip() == ""
    assert result["is_silent"] is True
    assert result["no_speech_prob"] == 1.0


def test_transcribe_faint_white_noise(transcriber, tmp_path):
    # Generate 1.5 seconds of faint Gaussian noise (amplitude well below min_rms = 0.003)
    sr = 16000
    duration = 1.5
    n_samples = int(sr * duration)
    # std = 10 out of 32767 -> rms approx 0.0003
    np.random.seed(42)
    faint_noise = np.random.normal(0, 10, n_samples).astype(np.int16)

    noise_file = str(tmp_path / "faint_noise.wav")
    with wave.open(noise_file, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(faint_noise.tobytes())

    rms, dur = analyze_audio_source(noise_file)
    assert rms < 0.002, f"Noise RMS {rms} is too high for faint noise test"
    assert np.isclose(dur, 1.5, atol=0.1)

    result = transcriber.transcribe(noise_file)
    assert result["text"].strip() == "", f"Expected silence/no command, got: '{result['text']}'"
    assert result["is_silent"] is True
    assert result["no_speech_prob"] == 1.0
