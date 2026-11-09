"""
Unit tests for ASR speech transcription.
Verifies faster-whisper tiny.en transcriber on real audio and synthetic silence.
"""

import os
import pytest
import numpy as np
import wave

from src.asr.transcriber import WhisperASR


@pytest.fixture
def transcriber():
    return WhisperASR(model_size="tiny.en", compute_type="int8")


def test_transcribe_sample_wav(transcriber):
    sample_path = "results/sample_speech.wav"
    assert os.path.exists(sample_path), f"Audio fixture missing: {sample_path}"
    
    result = transcriber.transcribe(sample_path)
    assert result["text"] != ""
    assert result["latency_sec"] > 0
    # The speech in sample_speech.wav was "Reach to the left."
    text_lower = result["text"].lower()
    assert "reach" in text_lower or "left" in text_lower


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
