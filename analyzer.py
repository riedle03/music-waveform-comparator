import numpy as np
import librosa
from io import BytesIO

WAVEFORM_SAMPLES = 2000


def load_audio(file_bytes: bytes) -> tuple[np.ndarray, int]:
    y, sr = librosa.load(BytesIO(file_bytes), sr=None, mono=True)
    return y, sr


def get_waveform(y: np.ndarray, n_samples: int = WAVEFORM_SAMPLES) -> np.ndarray:
    step = max(1, len(y) // n_samples)
    return y[::step][:n_samples]


def rms_similarity(y1: np.ndarray, y2: np.ndarray) -> float:
    rms1 = librosa.feature.rms(y=y1)[0]
    rms2 = librosa.feature.rms(y=y2)[0]
    min_len = min(len(rms1), len(rms2))
    score = 1.0 - np.mean(np.abs(rms1[:min_len] - rms2[:min_len]))
    return float(np.clip(score, 0.0, 1.0)) * 100.0
