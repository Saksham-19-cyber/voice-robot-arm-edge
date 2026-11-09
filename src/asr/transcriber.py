"""
ASR module powered by faster-whisper (tiny.en, int8 on CPU).
Optimized for low-latency voice command transcription on edge and cloud CPU.
Includes silence/RMS energy guard and hallucination suppression.
"""

import io
import os
import time
import tempfile
import numpy as np
from typing import Dict, Any, Union, Tuple
from faster_whisper import WhisperModel

HALLUCINATIONS = {
    "you", "you.", "thank you", "thank you.", "subtitles by", "bye", "bye.", "okay", "okay."
}


def analyze_audio_source(audio_source: Union[str, bytes, io.BytesIO, np.ndarray]) -> Tuple[float, float]:
    """
    Computes root-mean-square (RMS) energy normalized to [0.0, 1.0] and duration in seconds.
    Used to detect silence/faint background noise before ASR decoding.
    """
    rms = 1.0
    duration = 0.0
    try:
        if isinstance(audio_source, np.ndarray):
            arr = audio_source.astype(np.float32)
            if np.issubdtype(audio_source.dtype, np.integer):
                max_val = float(np.iinfo(audio_source.dtype).max)
                arr = arr / max_val
            rms = float(np.sqrt(np.mean(arr ** 2))) if arr.size > 0 else 0.0
            duration = len(arr) / 16000.0
            return rms, duration

        raw_bytes = None
        if isinstance(audio_source, (bytes, bytearray)):
            raw_bytes = bytes(audio_source)
        elif isinstance(audio_source, io.BytesIO):
            raw_bytes = audio_source.getvalue()

        if raw_bytes is not None:
            import wave
            try:
                with wave.open(io.BytesIO(raw_bytes), "rb") as wf:
                    sampwidth = wf.getsampwidth()
                    framerate = wf.getframerate() or 16000
                    n_frames = wf.getnframes()
                    duration = n_frames / float(framerate)
                    frames = wf.readframes(n_frames)
                    if sampwidth == 2:
                        samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                    elif sampwidth == 1:
                        samples = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128) / 128.0
                    elif sampwidth == 4:
                        samples = np.frombuffer(frames, dtype=np.float32)
                    else:
                        samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                    rms = float(np.sqrt(np.mean(samples ** 2))) if len(samples) > 0 else 0.0
                    return rms, duration
            except Exception:
                pass

        if isinstance(audio_source, str) and os.path.exists(audio_source):
            import wave
            try:
                with wave.open(audio_source, "rb") as wf:
                    sampwidth = wf.getsampwidth()
                    framerate = wf.getframerate() or 16000
                    n_frames = wf.getnframes()
                    duration = n_frames / float(framerate)
                    frames = wf.readframes(n_frames)
                    if sampwidth == 2:
                        samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                    elif sampwidth == 1:
                        samples = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128) / 128.0
                    elif sampwidth == 4:
                        samples = np.frombuffer(frames, dtype=np.float32)
                    else:
                        samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                    rms = float(np.sqrt(np.mean(samples ** 2))) if len(samples) > 0 else 0.0
                    return rms, duration
            except Exception:
                pass
    except Exception:
        pass
    return rms, duration


class WhisperASR:
    """Wrapper for faster-whisper quantized edge speech recognition."""

    def __init__(
        self,
        model_size: str = "tiny.en",
        device: str = "cpu",
        compute_type: str = "int8",
        cpu_threads: int = 1,
        min_rms: float = 0.003,
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.cpu_threads = cpu_threads
        self.min_rms = min_rms
        print(f"Loading faster-whisper model ({model_size}, {device}, {compute_type}, cpu_threads={cpu_threads})...", flush=True)
        t0 = time.time()
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
        )
        self.load_time = round(time.time() - t0, 3)
        print(f"Loaded faster-whisper in {self.load_time}s", flush=True)

    def transcribe(self, audio_source: Union[str, bytes, io.BytesIO, np.ndarray]) -> Dict[str, Any]:
        """
        Transcribes audio from a filepath, raw audio bytes, BytesIO, or numpy array.
        Returns dictionary containing transcript, latency, duration, and RTF.
        """
        t0 = time.time()

        # Check energy and duration before invoking ASR
        energy_rms, audio_duration = analyze_audio_source(audio_source)
        if energy_rms < self.min_rms:
            return {
                "text": "",
                "latency_sec": round(time.time() - t0, 4),
                "audio_duration_sec": round(audio_duration, 3),
                "rtf": 0.0,
                "language": "en",
                "language_probability": 0.0,
                "is_silent": True,
                "energy_rms": round(energy_rms, 6),
                "no_speech_prob": 1.0,
            }

        temp_audio_file = None
        try:
            if isinstance(audio_source, (bytes, bytearray)):
                temp_audio_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                temp_audio_file.write(audio_source)
                temp_audio_file.flush()
                temp_audio_file.close()
                input_path = temp_audio_file.name
            elif isinstance(audio_source, io.BytesIO):
                temp_audio_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                temp_audio_file.write(audio_source.getvalue())
                temp_audio_file.flush()
                temp_audio_file.close()
                input_path = temp_audio_file.name
            else:
                input_path = audio_source

            segments, info = self.model.transcribe(
                input_path,
                beam_size=1,
                language="en",
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=400),
            )

            text_segments = []
            max_no_speech_prob = 0.0
            for s in segments:
                nsp = getattr(s, "no_speech_prob", 0.0)
                if nsp > max_no_speech_prob:
                    max_no_speech_prob = nsp
                if nsp > 0.6:
                    continue
                seg_text = s.text.strip()
                if seg_text.lower() in HALLUCINATIONS:
                    continue
                if seg_text:
                    text_segments.append(seg_text)

            full_text = " ".join(text_segments).strip()
            if full_text.lower() in HALLUCINATIONS:
                full_text = ""

            latency = time.time() - t0
            duration = getattr(info, "duration", audio_duration)
            rtf = (latency / duration) if duration > 0 else 0.0

            return {
                "text": full_text,
                "latency_sec": round(latency, 4),
                "audio_duration_sec": round(duration, 3),
                "rtf": round(rtf, 4),
                "language": getattr(info, "language", "en"),
                "language_probability": round(getattr(info, "language_probability", 1.0), 3),
                "is_silent": False,
                "energy_rms": round(energy_rms, 6),
                "no_speech_prob": round(max_no_speech_prob, 3),
            }
        except Exception as e:
            return {
                "text": "",
                "latency_sec": round(time.time() - t0, 4),
                "audio_duration_sec": 0.0,
                "rtf": 0.0,
                "error": str(e),
                "is_silent": False,
            }
        finally:
            if temp_audio_file is not None:
                try:
                    os.unlink(temp_audio_file.name)
                except Exception:
                    pass
