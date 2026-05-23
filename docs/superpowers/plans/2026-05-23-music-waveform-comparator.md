# 노래 파형 비교기 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 학생이 원곡 MP3와 녹음본 MP3를 업로드하면 파형 3종 시각화와 단계적 유사도 점수를 보여주는 Streamlit 웹 앱을 구축한다.

**Architecture:** Streamlit 단일 앱으로 UI + 분석을 모두 처리한다. `analyzer.py`가 librosa 분석 로직을 담당하고, `app.py`가 Streamlit UI를 담당한다. 분석 함수는 `@st.cache_data`로 캐시된다.

**Tech Stack:** Python 3.11+, Streamlit, librosa, ffmpeg, Plotly, numpy, pytest

---

## 파일 구조

```
음악프로젝트/
├── app.py                   ← Streamlit UI (업로드, 시각화, 점수 표시)
├── analyzer.py              ← 순수 분석 함수 (librosa, numpy만 의존)
├── requirements.txt         ← Python 패키지 목록
├── packages.txt             ← 시스템 패키지 (ffmpeg)
├── .streamlit/
│   └── config.toml          ← Streamlit 테마 설정
└── tests/
    └── test_analyzer.py     ← analyzer.py 단위 테스트
```

---

## Task 1: 프로젝트 초기 설정

**Files:**
- Create: `requirements.txt`
- Create: `packages.txt`
- Create: `.streamlit/config.toml`

- [ ] **Step 1: requirements.txt 작성**

```
streamlit>=1.35.0
librosa>=0.10.0
plotly>=5.20.0
numpy>=1.26.0
scipy>=1.13.0
soundfile>=0.12.0
pytest>=8.0.0
```

- [ ] **Step 2: packages.txt 작성** (Streamlit Cloud에서 ffmpeg 자동 설치)

```
ffmpeg
```

- [ ] **Step 3: .streamlit/config.toml 작성**

```toml
[theme]
base = "light"
primaryColor = "#3B82F6"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F8FAFC"
textColor = "#1E293B"
font = "sans serif"
```

- [ ] **Step 4: 가상환경 생성 및 패키지 설치**

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

예상 출력: `Successfully installed streamlit-x.x.x librosa-x.x.x ...`

- [ ] **Step 5: Streamlit 설치 확인**

```bash
streamlit --version
```

예상 출력: `Streamlit, version 1.35.x`

---

## Task 2: analyzer.py — 오디오 로딩 + 파형 추출

**Files:**
- Create: `analyzer.py`
- Create: `tests/test_analyzer.py`

- [ ] **Step 1: 테스트 먼저 작성**

`tests/test_analyzer.py`:

```python
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
```

- [ ] **Step 2: 테스트 실행 — 실패 확인**

```bash
pytest tests/test_analyzer.py -v
```

예상 출력: `FAILED ... ModuleNotFoundError: No module named 'analyzer'`

- [ ] **Step 3: analyzer.py 구현 (로딩 + 파형)**

`analyzer.py`:

```python
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
```

- [ ] **Step 4: 테스트 실행 — 통과 확인**

```bash
pytest tests/test_analyzer.py -v
```

예상 출력: `PASSED test_load_audio_returns_array_and_sr`, `PASSED test_get_waveform_length`

- [ ] **Step 5: 커밋**

```bash
git init
git add analyzer.py tests/test_analyzer.py requirements.txt packages.txt .streamlit/config.toml
git commit -m "feat: 프로젝트 초기 설정 및 오디오 로딩/파형 추출"
```

---

## Task 3: analyzer.py — RMS 유사도 (1단계)

**Files:**
- Modify: `analyzer.py`
- Modify: `tests/test_analyzer.py`

- [ ] **Step 1: 테스트 추가**

`tests/test_analyzer.py` 하단에 추가:

```python
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
```

- [ ] **Step 2: 테스트 실행 — 실패 확인**

```bash
pytest tests/test_analyzer.py::test_rms_similarity_identical_files -v
```

예상 출력: `FAILED ... ImportError: cannot import name 'rms_similarity'`

- [ ] **Step 3: rms_similarity 구현**

`analyzer.py`에 추가:

```python
def rms_similarity(y1: np.ndarray, y2: np.ndarray) -> float:
    rms1 = librosa.feature.rms(y=y1)[0]
    rms2 = librosa.feature.rms(y=y2)[0]
    min_len = min(len(rms1), len(rms2))
    score = 1.0 - np.mean(np.abs(rms1[:min_len] - rms2[:min_len]))
    return float(np.clip(score, 0.0, 1.0)) * 100.0
```

- [ ] **Step 4: 테스트 실행 — 통과 확인**

```bash
pytest tests/test_analyzer.py -v
```

예상 출력: 모든 테스트 `PASSED`

- [ ] **Step 5: 커밋**

```bash
git add analyzer.py tests/test_analyzer.py
git commit -m "feat: RMS 진폭 유사도 분석 (1단계)"
```

---

## Task 4: app.py — UI 뼈대 (CSS + Footer + 업로드)

**Files:**
- Create: `app.py`

- [ ] **Step 1: app.py 작성**

```python
import streamlit as st
import numpy as np
from analyzer import load_audio, get_waveform, rms_similarity

# ── 페이지 설정 ──────────────────────────────────────────
st.set_page_config(
    page_title="노래 파형 비교기",
    page_icon="🎵",
    layout="wide",
)

# ── Pretendard 폰트 + 전역 스타일 ────────────────────────
st.markdown("""
<link rel="stylesheet"
  href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css"/>
<style>
  html, body, [class*="css"] {
    font-family: 'Pretendard', sans-serif !important;
  }
  h1 { font-size: 2.4rem !important; font-weight: 800 !important; }
  h2 { font-size: 1.6rem !important; font-weight: 700 !important; }
  .subtitle {
    color: #64748B;
    font-size: 1.1rem;
    margin-top: -0.8rem;
    margin-bottom: 1.5rem;
  }
  .score-box {
    background: #F1F5F9;
    border-radius: 12px;
    padding: 1.2rem 1.6rem;
    margin-bottom: 0.8rem;
  }
  .score-label { font-size: 1rem; color: #475569; font-weight: 600; }
  .score-value { font-size: 2rem; font-weight: 800; color: #1E293B; }
  .badge-pending {
    background: #E2E8F0;
    color: #94A3B8;
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.75rem;
    margin-left: 8px;
  }
  footer { visibility: hidden; }
  .custom-footer {
    text-align: center;
    color: #94A3B8;
    font-size: 0.85rem;
    padding: 2rem 0 1rem;
    border-top: 1px solid #E2E8F0;
    margin-top: 3rem;
  }
</style>
""", unsafe_allow_html=True)

# ── 헤더 ─────────────────────────────────────────────────
st.markdown("## ♩ 노래 파형 비교기")
st.markdown('<p class="subtitle">원곡과 녹음본의 파형을 분석합니다</p>',
            unsafe_allow_html=True)

# ── 파일 업로드 ───────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    original_file = st.file_uploader("원곡 MP3", type=["mp3", "wav", "m4a"],
                                     key="original")
with col2:
    recording_file = st.file_uploader("녹음본 MP3", type=["mp3", "wav", "m4a"],
                                      key="recording")

# ── Footer ────────────────────────────────────────────────
st.markdown(
    '<div class="custom-footer">© 2026 이대형 · riedel@e-mirim.hs.kr</div>',
    unsafe_allow_html=True,
)
```

- [ ] **Step 2: 앱 실행 확인**

```bash
streamlit run app.py
```

브라우저에서 `http://localhost:8501` 열어 확인:
- 타이틀 "노래 파형 비교기" 표시
- 업로드 위젯 2개 표시
- Footer 표시
- Pretendard 폰트 적용 확인

- [ ] **Step 3: 커밋**

```bash
git add app.py
git commit -m "feat: UI 뼈대 — 헤더, 업로드 위젯, Footer, Pretendard 폰트"
```

---

## Task 5: app.py — 파형 시각화 3탭

**Files:**
- Modify: `app.py`

- [ ] **Step 1: 파형 시각화 함수 추가**

`app.py`의 import 블록에 추가:

```python
import plotly.graph_objects as go
```

`app.py`에 함수 추가 (파일 업로드 코드 위에):

```python
ORIGINAL_COLOR = "#3B82F6"
RECORDING_COLOR = "#F97316"


def make_filled_fig(w1: np.ndarray, w2: np.ndarray) -> go.Figure:
    t1 = np.linspace(0, 1, len(w1))
    t2 = np.linspace(0, 1, len(w2))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t1, y=w1, name="원곡",
        fill="tozeroy", fillcolor="rgba(59,130,246,0.25)",
        line=dict(color=ORIGINAL_COLOR, width=1),
    ))
    fig.add_trace(go.Scatter(
        x=t2, y=w2, name="녹음본",
        fill="tozeroy", fillcolor="rgba(249,115,22,0.25)",
        line=dict(color=RECORDING_COLOR, width=1),
    ))
    fig.update_layout(_chart_layout("채움 파형"))
    return fig


def make_bar_fig(w1: np.ndarray, w2: np.ndarray) -> go.Figure:
    n = 300
    s1 = max(1, len(w1) // n)
    s2 = max(1, len(w2) // n)
    b1 = np.abs(w1[::s1][:n])
    b2 = np.abs(w2[::s2][:n])
    t1 = np.linspace(0, 1, len(b1))
    t2 = np.linspace(0, 1, len(b2))
    fig = go.Figure()
    fig.add_trace(go.Bar(x=t1, y=b1, name="원곡",
                         marker_color=ORIGINAL_COLOR, opacity=0.7))
    fig.add_trace(go.Bar(x=t2, y=b2, name="녹음본",
                         marker_color=RECORDING_COLOR, opacity=0.7))
    fig.update_layout(_chart_layout("막대 파형"), barmode="overlay")
    return fig


def make_line_fig(w1: np.ndarray, w2: np.ndarray) -> go.Figure:
    t1 = np.linspace(0, 1, len(w1))
    t2 = np.linspace(0, 1, len(w2))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t1, y=w1, name="원곡",
        line=dict(color=ORIGINAL_COLOR, width=1),
    ))
    fig.add_trace(go.Scatter(
        x=t2, y=w2, name="녹음본",
        line=dict(color=RECORDING_COLOR, width=1, dash="dot"),
    ))
    fig.update_layout(_chart_layout("라인 파형"))
    return fig


def _chart_layout(title: str) -> dict:
    return dict(
        title=title,
        xaxis_title="시간 (정규화)",
        yaxis_title="진폭",
        height=300,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", y=1.1),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(family="Pretendard, sans-serif"),
    )
```

- [ ] **Step 2: 파형 탭 렌더링 코드 추가**

파일 업로드 코드 아래, Footer 위에 삽입:

```python
@st.cache_data(show_spinner=False)
def cached_load(file_bytes: bytes):
    y, sr = load_audio(file_bytes)
    w = get_waveform(y)
    return y, sr, w

if original_file and recording_file:
    with st.spinner("파형을 분석하는 중..."):
        orig_bytes = original_file.read()
        rec_bytes = recording_file.read()
        y1, sr1, w1 = cached_load(orig_bytes)
        y2, sr2, w2 = cached_load(rec_bytes)

    st.markdown("### 파형 비교")
    tab_fill, tab_bar, tab_line = st.tabs(["채움 파형", "막대 파형", "라인 파형"])
    with tab_fill:
        st.plotly_chart(make_filled_fig(w1, w2), use_container_width=True)
    with tab_bar:
        st.plotly_chart(make_bar_fig(w1, w2), use_container_width=True)
    with tab_line:
        st.plotly_chart(make_line_fig(w1, w2), use_container_width=True)
```

- [ ] **Step 3: 앱 실행 후 MP3 두 개 업로드해서 탭 동작 확인**

```bash
streamlit run app.py
```

확인 항목:
- 채움 파형: 파랑/주황 반투명 면 겹침
- 막대 파형: 블록 막대 두 색
- 라인 파형: 실선(원곡) + 점선(녹음본)
- 탭 전환 동작

- [ ] **Step 4: 커밋**

```bash
git add app.py
git commit -m "feat: 파형 시각화 3탭 (채움/막대/라인)"
```

---

## Task 6: app.py — RMS 점수 표시 → 1단계 MVP 완성

**Files:**
- Modify: `app.py`

- [ ] **Step 1: 점수 표시 코드 추가**

파형 탭 코드 아래, Footer 위에 삽입:

```python
    st.markdown("### 유사도 분석")

    rms_score = rms_similarity(y1, y2)
    overall = rms_score  # 1단계: RMS만

    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.markdown(f"""
        <div class="score-box">
          <div class="score-label">전체 유사도</div>
          <div class="score-value">{overall:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with col_b:
        st.markdown(f"""
        <div class="score-box">
          <div class="score-label">진폭 일치도</div>
          <div class="score-value">{rms_score:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with col_c:
        st.markdown("""
        <div class="score-box">
          <div class="score-label">음정 일치도
            <span class="badge-pending">준비 중</span>
          </div>
          <div class="score-value" style="color:#CBD5E1">—</div>
        </div>""", unsafe_allow_html=True)
    with col_d:
        st.markdown("""
        <div class="score-box">
          <div class="score-label">음색 일치도
            <span class="badge-pending">준비 중</span>
          </div>
          <div class="score-value" style="color:#CBD5E1">—</div>
        </div>""", unsafe_allow_html=True)

    st.progress(int(overall))
```

- [ ] **Step 2: 앱 실행 후 점수 확인**

```bash
streamlit run app.py
```

확인 항목:
- 전체 유사도 숫자 표시
- 진폭 일치도 숫자 표시
- 음정/음색은 "준비 중" 뱃지 + "—" 표시
- 프로그레스 바 렌더링

- [ ] **Step 3: 커밋 (1단계 MVP 완성)**

```bash
git add app.py
git commit -m "feat: RMS 점수 표시 — 1단계 MVP 완성"
```

---

## Task 7: analyzer.py — 피치 유사도 (2단계)

**Files:**
- Modify: `analyzer.py`
- Modify: `tests/test_analyzer.py`

- [ ] **Step 1: 테스트 추가**

`tests/test_analyzer.py` 하단에 추가:

```python
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
```

- [ ] **Step 2: 테스트 실행 — 실패 확인**

```bash
pytest tests/test_analyzer.py::test_pitch_similarity_identical -v
```

예상 출력: `FAILED ... ImportError: cannot import name 'pitch_similarity'`

- [ ] **Step 3: pitch_similarity 구현**

`analyzer.py`에 추가:

```python
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
```

- [ ] **Step 4: 테스트 실행 — 통과 확인**

```bash
pytest tests/test_analyzer.py -v
```

예상 출력: 모든 테스트 `PASSED`

- [ ] **Step 5: 커밋**

```bash
git add analyzer.py tests/test_analyzer.py
git commit -m "feat: 피치(음높이) 유사도 분석 (2단계)"
```

---

## Task 8: app.py — 피치 점수 UI (2단계)

**Files:**
- Modify: `app.py`

- [ ] **Step 1: import 수정**

`app.py` import 줄 수정:

```python
from analyzer import load_audio, get_waveform, rms_similarity, pitch_similarity
```

- [ ] **Step 2: 분석 블록 수정 — 피치 점수 추가 + 전체 유사도 가중치 반영**

Task 6에서 작성한 `overall = rms_score` 줄과 점수 표시 블록 전체를 아래로 교체:

```python
    rms_score = rms_similarity(y1, y2)

    with st.spinner("음정을 분석하는 중..."):
        p_score = pitch_similarity(y1, sr1, y2, sr2)

    # 2단계 가중치: RMS 20% + Pitch 80% (MFCC 추가 전 임시)
    overall = rms_score * 0.2 + p_score * 0.8

    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.markdown(f"""
        <div class="score-box">
          <div class="score-label">전체 유사도</div>
          <div class="score-value">{overall:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with col_b:
        st.markdown(f"""
        <div class="score-box">
          <div class="score-label">진폭 일치도</div>
          <div class="score-value">{rms_score:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with col_c:
        st.markdown(f"""
        <div class="score-box">
          <div class="score-label">음정 일치도</div>
          <div class="score-value">{p_score:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with col_d:
        st.markdown("""
        <div class="score-box">
          <div class="score-label">음색 일치도
            <span class="badge-pending">준비 중</span>
          </div>
          <div class="score-value" style="color:#CBD5E1">—</div>
        </div>""", unsafe_allow_html=True)

    st.progress(int(overall))
```

- [ ] **Step 3: 앱 실행 후 확인**

```bash
streamlit run app.py
```

확인 항목:
- 음정 일치도 숫자 표시 (2-4초 소요)
- 전체 유사도가 RMS 20% + Pitch 80% 가중 평균으로 계산됨
- 음색 일치도만 "준비 중" 상태

- [ ] **Step 4: 커밋**

```bash
git add app.py
git commit -m "feat: 피치 점수 UI — 2단계 완성"
```

---

## Task 9: analyzer.py — MFCC + DTW 유사도 (3단계)

**Files:**
- Modify: `analyzer.py`
- Modify: `tests/test_analyzer.py`

- [ ] **Step 1: 테스트 추가**

`tests/test_analyzer.py` 하단에 추가:

```python
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
```

- [ ] **Step 2: 테스트 실행 — 실패 확인**

```bash
pytest tests/test_analyzer.py::test_mfcc_dtw_identical -v
```

예상 출력: `FAILED ... ImportError: cannot import name 'mfcc_dtw_similarity'`

- [ ] **Step 3: mfcc_dtw_similarity 구현**

`analyzer.py` import 블록에 추가:

```python
from scipy.spatial.distance import cdist
```

`analyzer.py`에 함수 추가:

```python
def mfcc_dtw_similarity(y1: np.ndarray, sr1: int,
                        y2: np.ndarray, sr2: int) -> float:
    mfcc1 = librosa.feature.mfcc(y=y1, sr=sr1, n_mfcc=13)
    mfcc2 = librosa.feature.mfcc(y=y2, sr=sr2, n_mfcc=13)
    C = cdist(mfcc1.T, mfcc2.T, metric="cosine")
    D, wp = librosa.sequence.dtw(C=C, backtrack=True)
    score = 1.0 / (1.0 + D[-1, -1] / max(len(wp), 1))
    return float(np.clip(score, 0.0, 1.0)) * 100.0
```

- [ ] **Step 4: 테스트 실행 — 통과 확인**

```bash
pytest tests/test_analyzer.py -v
```

예상 출력: 모든 테스트 `PASSED`

- [ ] **Step 5: 커밋**

```bash
git add analyzer.py tests/test_analyzer.py
git commit -m "feat: MFCC+DTW 음색/박자 유사도 분석 (3단계)"
```

---

## Task 10: app.py — 종합 점수 UI (3단계 완성)

**Files:**
- Modify: `app.py`

- [ ] **Step 1: import 수정**

```python
from analyzer import (load_audio, get_waveform, rms_similarity,
                      pitch_similarity, mfcc_dtw_similarity)
```

- [ ] **Step 2: 분석 + 점수 블록 최종 교체**

Task 8의 분석 블록 전체를 아래로 교체:

```python
    rms_score = rms_similarity(y1, y2)

    with st.spinner("음정을 분석하는 중..."):
        p_score = pitch_similarity(y1, sr1, y2, sr2)

    with st.spinner("음색과 박자를 분석하는 중..."):
        mfcc_score = mfcc_dtw_similarity(y1, sr1, y2, sr2)

    # 최종 가중치: RMS 20% + Pitch 40% + MFCC/DTW 40%
    overall = rms_score * 0.2 + p_score * 0.4 + mfcc_score * 0.4

    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.markdown(f"""
        <div class="score-box">
          <div class="score-label">전체 유사도</div>
          <div class="score-value">{overall:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with col_b:
        st.markdown(f"""
        <div class="score-box">
          <div class="score-label">진폭 일치도</div>
          <div class="score-value">{rms_score:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with col_c:
        st.markdown(f"""
        <div class="score-box">
          <div class="score-label">음정 일치도</div>
          <div class="score-value">{p_score:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with col_d:
        st.markdown(f"""
        <div class="score-box">
          <div class="score-label">음색 일치도</div>
          <div class="score-value">{mfcc_score:.1f}%</div>
        </div>""", unsafe_allow_html=True)

    st.progress(int(overall))
```

- [ ] **Step 3: 앱 실행 후 3단계 전체 동작 확인**

```bash
streamlit run app.py
```

확인 항목:
- 4개 점수 모두 숫자 표시
- 전체 유사도 = RMS×0.2 + Pitch×0.4 + MFCC×0.4
- 3종 파형 탭 정상 동작
- Footer 표시

- [ ] **Step 4: 커밋**

```bash
git add app.py
git commit -m "feat: 종합 점수 UI — 3단계 완성"
```

---

## Task 11: 배포 — Streamlit Cloud

**Files:**
- (기존 파일 확인, 추가 없음)

- [ ] **Step 1: GitHub 저장소 생성 후 푸시**

```bash
git remote add origin https://github.com/<your-username>/music-waveform-comparator.git
git branch -M main
git push -u origin main
```

- [ ] **Step 2: Streamlit Cloud 연결**

1. [share.streamlit.io](https://share.streamlit.io) 접속
2. "New app" 클릭
3. GitHub 저장소 선택
4. Main file path: `app.py`
5. "Deploy!" 클릭

- [ ] **Step 3: 배포 후 동작 확인**

발급된 URL (예: `https://<username>-music-waveform.streamlit.app`)에서:
- 파일 업로드 정상 동작
- 파형 시각화 3탭 확인
- 4개 점수 확인
- 모바일 브라우저에서도 확인

- [ ] **Step 4: 최종 커밋**

```bash
git add .
git commit -m "chore: 배포 완료"
git push
```
