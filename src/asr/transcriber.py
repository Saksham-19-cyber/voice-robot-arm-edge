"""
ASR module powered by faster-whisper (tiny.en, int8 on CPU).
Optimized for low-latency voice command transcription on edge and cloud CPU.
"""

import io
import time
import tempfile
import numpy as np
from typing import Dict, Any, Union
from faster_whisper import WhisperModel


class WhisperASR:
    """Wrapper for faster-whisper quantized edge speech recognition."""

    def __init__(self, model_size: str = "tiny.en", device: str = "cpu", compute_type: str = "int8"):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        print(f"Loading faster-whisper model ({model_size}, {device}, {compute_type})...", flush=True)
        t0 = time.time()
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.load_time = round(time.time() - t0, 3)
        print(f"Loaded faster-whisper in {self.load_time}s", flush=True)

    def transcribe(self, audio_source: Union[str, bytes, io.BytesIO, np.ndarray]) -> Dict[str, Any]:
        """
        Transcribes audio from a filepath, raw audio bytes, BytesIO, or numpy array.
        Returns dictionary containing transcript, latency, duration, and RTF.
        """
        t0 = time.time()

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

            text_segments = [s.text.strip() for s in segments]
            full_text = " ".join(text_segments).strip()

            latency = time.time() - t0
            duration = getattr(info, "duration", 0.0)
            rtf = (latency / duration) if duration > 0 else 0.0

            return {
                "text": full_text,
                "latency_sec": round(latency, 4),
                "audio_duration_sec": round(duration, 3),
                "rtf": round(rtf, 4),
                "language": getattr(info, "language", "en"),
                "language_probability": round(getattr(info, "language_probability", 1.0), 3),
            }
        except Exception as e:
            return {
                "text": "",
                "latency_sec": round(time.time() - t0, 4),
                "audio_duration_sec": 0.0,
                "rtf": 0.0,
                "error": str(e),
            }
        finally:
            if temp_audio_file is not None:
                try:
                    import os
                    os.unlink(temp_audio_file.name)
                except Exception:
                    pass

