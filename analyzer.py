import numpy as np
import librosa
from io import BytesIO
from scipy.spatial.distance import cdist

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


def pitch_similarity(y1: np.ndarray, sr1: int,
                     y2: np.ndarray, sr2: int) -> float:
    pitch1 = librosa.yin(y1, fmin=80, fmax=1000, sr=sr1)
    pitch2 = librosa.yin(y2, fmin=80, fmax=1000, sr=sr2)
    voiced1 = pitch1[pitch1 > 0]
    voiced2 = pitch2[pitch2 > 0]
    min_len = min(len(voiced1), len(voiced2))
    if min_len == 0:
        return 0.0
    diff = (np.abs(voiced1[:min_len] - voiced2[:min_len])
            / (voiced1[:min_len] + voiced2[:min_len] + 1e-6) * 2)
    score = 1.0 - np.mean(np.clip(diff, 0.0, 1.0))
    return float(np.clip(score, 0.0, 1.0)) * 100.0


def mfcc_dtw_similarity(y1: np.ndarray, sr1: int,
                        y2: np.ndarray, sr2: int) -> float:
    mfcc1 = librosa.feature.mfcc(y=y1, sr=sr1, n_mfcc=13)
    mfcc2 = librosa.feature.mfcc(y=y2, sr=sr2, n_mfcc=13)
    C = cdist(mfcc1.T, mfcc2.T, metric="cosine")
    D, wp = librosa.sequence.dtw(C=C, backtrack=True)
    score = 1.0 / (1.0 + D[-1, -1] / max(len(wp), 1))
    return float(np.clip(score, 0.0, 1.0)) * 100.0
