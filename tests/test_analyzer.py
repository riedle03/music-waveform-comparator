import numpy as np
import pytest
from io import BytesIO
import soundfile as sf
from analyzer import load_audio, get_waveform

def make_sine_mp3_bytes(freq=440, duration=2, sr=22050) -> bytes:
    """테스트용 합성 사인파 오디오를 WAV bytes로 반환 (ffmpeg 없이 테스트 가능)"""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = (np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
    buf = BytesIO()
    sf.write(buf, y, sr, format='WAV', subtype='PCM_16')
    buf.seek(0)
    return buf.read()

def test_load_audio_returns_array_and_sr():
    audio_bytes = make_sine_mp3_bytes()
    y, sr = load_audio(audio_bytes)
    assert isinstance(y, np.ndarray)
    assert sr == 22050
    assert len(y) > 0

def test_get_waveform_length():
    audio_bytes = make_sine_mp3_bytes()
    y, sr = load_audio(audio_bytes)
    waveform = get_waveform(y, n_samples=500)
    assert len(waveform) <= 500
