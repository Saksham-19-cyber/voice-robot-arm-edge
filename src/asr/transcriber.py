"""faster-whisper ASR transcriber."""





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


def _extract_wav_samples_and_duration(raw_bytes: bytes) -> Tuple[np.ndarray, float]:
    """Parse RIFF/WAV audio bytes into float32 array in [-1.0, 1.0] and duration.
    
    Robustly handles PCM int16, PCM int32 (format code 1), and IEEE float32 (format code 3)
    without relying on external soundfile library or python wave limitations on float32.
    """
    import struct
    if len(raw_bytes) < 12 or raw_bytes[:4] != b"RIFF" or raw_bytes[8:12] != b"WAVE":
        # Fallback to standard wave if not standard RIFF header
        import wave
        with wave.open(io.BytesIO(raw_bytes), "rb") as wf:
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate() or 16000
            n_frames = wf.getnframes()
            duration = n_frames / float(framerate)
            frames = wf.readframes(n_frames)
            samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            return samples, duration

    offset = 12
    fmt_info = None
    data_bytes = None

    while offset + 8 <= len(raw_bytes):
        chunk_id = raw_bytes[offset:offset + 4]
        chunk_size = struct.unpack("<I", raw_bytes[offset + 4:offset + 8])[0]
        chunk_data_start = offset + 8
        chunk_data_end = chunk_data_start + chunk_size

        if chunk_id == b"fmt ":
            audio_format, num_channels, sample_rate, byte_rate, block_align, bits_per_sample = struct.unpack(
                "<HHIIHH", raw_bytes[chunk_data_start:chunk_data_start + 16]
            )
            fmt_info = {
                "audio_format": audio_format,
                "num_channels": num_channels,
                "sample_rate": sample_rate,
                "bits_per_sample": bits_per_sample,
            }
        elif chunk_id == b"data":
            data_bytes = raw_bytes[chunk_data_start:chunk_data_end]

        offset = chunk_data_end + (chunk_size % 2)

    if fmt_info is None or data_bytes is None:
        raise ValueError("Missing fmt or data chunk in WAV")

    audio_format = fmt_info["audio_format"]
    bits = fmt_info["bits_per_sample"]
    channels = fmt_info["num_channels"]
    sample_rate = fmt_info["sample_rate"]

    if bits == 16:
        samples = np.frombuffer(data_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    elif bits == 8:
        samples = (np.frombuffer(data_bytes, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif bits == 32:
        if audio_format == 3:  # IEEE float32
            samples = np.frombuffer(data_bytes, dtype=np.float32)
        else:  # PCM int32 (format 1)
            samples = np.frombuffer(data_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        samples = np.frombuffer(data_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)

    duration = len(samples) / float(sample_rate) if sample_rate > 0 else 0.0
    return samples, duration


def analyze_audio_source(audio_source: Union[str, bytes, io.BytesIO, np.ndarray]) -> Tuple[float, float]:
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
        elif isinstance(audio_source, str) and os.path.exists(audio_source):
            with open(audio_source, "rb") as f:
                raw_bytes = f.read()

        if raw_bytes is not None:
            try:
                samples, duration = _extract_wav_samples_and_duration(raw_bytes)
                rms = float(np.sqrt(np.mean(samples ** 2))) if len(samples) > 0 else 0.0
                return rms, duration
            except Exception:
                pass
    except Exception:
        pass
    return rms, duration


class WhisperASR:


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




        t0 = time.time()


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
