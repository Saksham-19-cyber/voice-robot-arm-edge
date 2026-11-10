"""Unit tests for ASR module."""








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

    text_lower = result["text"].lower()
    assert "reach" in text_lower or "left" in text_lower
    assert result["is_silent"] is False
    assert result["energy_rms"] > 0.01


def test_transcribe_silent_audio(transcriber, tmp_path):

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

    sr = 16000
    duration = 1.5
    n_samples = int(sr * duration)

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


def test_wav_formats_int16_int32_float32(tmp_path):
    """Test analyze_audio_source handles int16, int32, and float32 WAVs with consistent RMS."""
    import struct
    sr = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    sine = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    expected_rms = float(np.sqrt(np.mean(sine ** 2)))

    def make_wav_file(path: str, subtype: str):
        if subtype == "int16":
            audio_format = 1
            bits = 16
            data = (sine * 32767.0).astype(np.int16).tobytes()
        elif subtype == "int32":
            audio_format = 1
            bits = 32
            data = (sine * 2147483647.0).astype(np.int32).tobytes()
        elif subtype == "float32":
            audio_format = 3
            bits = 32
            data = sine.astype(np.float32).tobytes()

        block_align = 1 * (bits // 8)
        byte_rate = sr * block_align
        fmt_chunk = struct.pack("<4sIHHIIHH", b"fmt ", 16, audio_format, 1, sr, byte_rate, block_align, bits)
        data_chunk = struct.pack("<4sI", b"data", len(data)) + data
        riff = struct.pack("<4sI4s", b"RIFF", 4 + len(fmt_chunk) + len(data_chunk), b"WAVE")
        with open(path, "wb") as f:
            f.write(riff + fmt_chunk + data_chunk)

    p_int16 = str(tmp_path / "test_int16.wav")
    p_int32 = str(tmp_path / "test_int32.wav")
    p_float32 = str(tmp_path / "test_float32.wav")

    make_wav_file(p_int16, "int16")
    make_wav_file(p_int32, "int32")
    make_wav_file(p_float32, "float32")

    rms_16, dur_16 = analyze_audio_source(p_int16)
    rms_32, dur_32 = analyze_audio_source(p_int32)
    rms_fl, dur_fl = analyze_audio_source(p_float32)

    assert np.isclose(dur_16, 1.0, atol=0.05)
    assert np.isclose(dur_32, 1.0, atol=0.05)
    assert np.isclose(dur_fl, 1.0, atol=0.05)

    assert np.isclose(rms_16, expected_rms, atol=0.02), f"int16 RMS mismatch: {rms_16} vs {expected_rms}"
    assert np.isclose(rms_32, expected_rms, atol=0.02), f"int32 RMS mismatch: {rms_32} vs {expected_rms}"
    assert np.isclose(rms_fl, expected_rms, atol=0.02), f"float32 RMS mismatch: {rms_fl} vs {expected_rms}"
