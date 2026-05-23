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
