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

from analyzer import rms_similarity

def test_rms_similarity_identical_files():
    audio_bytes = make_sine_mp3_bytes(freq=440)
    y, sr = load_audio(audio_bytes)
    score = rms_similarity(y, y)
    assert 95.0 <= score <= 100.0

def test_rms_similarity_different_files():
    bytes1 = make_sine_mp3_bytes(freq=440, duration=2)
    bytes2 = make_sine_mp3_bytes(freq=880, duration=2)
    y1, _ = load_audio(bytes1)
    y2, _ = load_audio(bytes2)
    score = rms_similarity(y1, y2)
    assert 0.0 <= score <= 100.0

def test_rms_similarity_different_lengths():
    bytes1 = make_sine_mp3_bytes(duration=2)
    bytes2 = make_sine_mp3_bytes(duration=3)
    y1, _ = load_audio(bytes1)
    y2, _ = load_audio(bytes2)
    score = rms_similarity(y1, y2)
    assert 0.0 <= score <= 100.0

from analyzer import pitch_similarity

def test_pitch_similarity_identical():
    audio_bytes = make_sine_mp3_bytes(freq=440, duration=2)
    y, sr = load_audio(audio_bytes)
    score = pitch_similarity(y, sr, y, sr)
    assert 90.0 <= score <= 100.0

def test_pitch_similarity_range():
    bytes1 = make_sine_mp3_bytes(freq=440, duration=2)
    bytes2 = make_sine_mp3_bytes(freq=660, duration=2)
    y1, sr1 = load_audio(bytes1)
    y2, sr2 = load_audio(bytes2)
    score = pitch_similarity(y1, sr1, y2, sr2)
    assert 0.0 <= score <= 100.0

from analyzer import mfcc_dtw_similarity

def test_mfcc_dtw_identical():
    audio_bytes = make_sine_mp3_bytes(freq=440, duration=2)
    y, sr = load_audio(audio_bytes)
    score = mfcc_dtw_similarity(y, sr, y, sr)
    assert 90.0 <= score <= 100.0

def test_mfcc_dtw_range():
    bytes1 = make_sine_mp3_bytes(freq=440, duration=2)
    bytes2 = make_sine_mp3_bytes(freq=880, duration=2)
    y1, sr1 = load_audio(bytes1)
    y2, sr2 = load_audio(bytes2)
    score = mfcc_dtw_similarity(y1, sr1, y2, sr2)
    assert 0.0 <= score <= 100.0
