import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import librosa
from analyzer import (load_audio, get_waveform, rms_similarity,
                      pitch_similarity, mfcc_dtw_similarity)
from export import generate_pdf, generate_xlsx

# ── 세션 상태 초기화 ──────────────────────────────────────
if 'results' not in st.session_state:
    st.session_state.results = None

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
  .tab-desc {
    background: #F8FAFC;
    border-left: 3px solid #CBD5E1;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1.2rem;
    margin-top: 0.5rem;
    font-size: 0.9rem;
    color: #475569;
    line-height: 1.6;
  }
</style>
""", unsafe_allow_html=True)

# ── 헤더 ─────────────────────────────────────────────────
st.markdown("## ♩ 노래 파형 비교기")
st.markdown('<p class="subtitle">원곡과 녹음본의 파형을 분석합니다</p>',
            unsafe_allow_html=True)

ORIGINAL_COLOR = "#3B82F6"
RECORDING_COLOR = "#F97316"


def make_spectrogram_fig(y1: np.ndarray, sr1: int,
                          y2: np.ndarray, sr2: int) -> go.Figure:
    D1_raw = librosa.stft(y1, n_fft=2048, hop_length=512)
    D2_raw = librosa.stft(y2, n_fft=2048, hop_length=512)
    D1 = librosa.amplitude_to_db(np.abs(D1_raw), ref=np.max)
    D2 = librosa.amplitude_to_db(np.abs(D2_raw), ref=np.max)

    freqs = librosa.fft_frequencies(sr=sr1, n_fft=2048)
    times1 = librosa.frames_to_time(np.arange(D1.shape[1]), sr=sr1, hop_length=512)
    times2 = librosa.frames_to_time(np.arange(D2.shape[1]), sr=sr2, hop_length=512)

    freq_mask = freqs <= 4000

    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=("원곡", "녹음본"),
                        vertical_spacing=0.12)

    fig.add_trace(
        go.Heatmap(z=D1[freq_mask], x=times1, y=freqs[freq_mask],
                   colorscale='Viridis', showscale=False, name="원곡"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Heatmap(z=D2[freq_mask], x=times2, y=freqs[freq_mask],
                   colorscale='Plasma', showscale=True, name="녹음본"),
        row=2, col=1,
    )

    fig.update_yaxes(title_text="주파수 (Hz)")
    fig.update_xaxes(title_text="시간 (초)", row=2, col=1)
    fig.update_layout(
        height=500,
        margin=dict(l=0, r=60, t=60, b=0),
        font=dict(family="Pretendard, sans-serif"),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
    )
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


def make_energy_heatmap_fig(y1: np.ndarray, y2: np.ndarray) -> go.Figure:
    rms1 = librosa.feature.rms(y=y1, frame_length=2048, hop_length=512)[0]
    rms2 = librosa.feature.rms(y=y2, frame_length=2048, hop_length=512)[0]

    s1 = max(1, len(rms1) // 200)
    r1 = rms1[::s1][:200]
    s2 = max(1, len(rms2) // 200)
    r2 = rms2[::s2][:200]

    t1 = np.linspace(0, 1, len(r1))
    t2 = np.linspace(0, 1, len(r2))

    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=("원곡", "녹음본"),
                        vertical_spacing=0.12)

    fig.add_trace(
        go.Bar(x=t1, y=r1,
               marker=dict(color=r1, colorscale='Blues', showscale=False),
               name="원곡"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Bar(x=t2, y=r2,
               marker=dict(color=r2, colorscale='Oranges', showscale=False),
               name="녹음본"),
        row=2, col=1,
    )

    fig.update_layout(
        height=450,
        margin=dict(l=0, r=60, t=60, b=0),
        font=dict(family="Pretendard, sans-serif"),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
    )
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


# ── 파일 업로드 ───────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    original_file = st.file_uploader("원곡 MP3", type=["mp3", "wav", "m4a"],
                                     key="original")
with col2:
    recording_file = st.file_uploader("녹음본 MP3", type=["mp3", "wav", "m4a"],
                                      key="recording")


@st.cache_data(show_spinner=False)
def cached_load(file_bytes: bytes):
    y, sr = load_audio(file_bytes)
    w = get_waveform(y)
    return y, sr, w


files_ready = original_file is not None and recording_file is not None

analyze_clicked = st.button(
    "분석 시작",
    disabled=not files_ready,
    type="primary",
    use_container_width=True,
)

if analyze_clicked:
    orig_bytes = original_file.read()
    rec_bytes = recording_file.read()
    y1, sr1, w1 = cached_load(orig_bytes)
    y2, sr2, w2 = cached_load(rec_bytes)

    with st.spinner("진폭 분석 중..."):
        rms_score = rms_similarity(y1, y2)
    with st.spinner("음정 분석 중..."):
        p_score = pitch_similarity(y1, sr1, y2, sr2)
    with st.spinner("음색·박자 분석 중..."):
        mfcc_score = mfcc_dtw_similarity(y1, sr1, y2, sr2)

    overall = rms_score * 0.2 + p_score * 0.4 + mfcc_score * 0.4

    st.session_state.results = {
        'overall': overall,
        'rms_score': rms_score,
        'p_score': p_score,
        'mfcc_score': mfcc_score,
        'y1': y1, 'sr1': sr1, 'w1': w1,
        'y2': y2, 'sr2': sr2, 'w2': w2,
    }

# ── 결과 표시 ─────────────────────────────────────────────
if st.session_state.results:
    r = st.session_state.results

    st.markdown("### 파형 비교")
    tab1, tab2, tab3, tab4 = st.tabs(["스펙트로그램", "막대 파형", "에너지 히트맵", "라인 파형"])

    with tab1:
        st.plotly_chart(make_spectrogram_fig(r['y1'], r['sr1'], r['y2'], r['sr2']),
                        use_container_width=True)
        st.markdown("""<div class="tab-desc">
        <b>스펙트로그램</b>은 STFT(단시간 푸리에 변환, Short-Time Fourier Transform)를 사용해 오디오 신호를 시간-주파수 2D 공간으로 변환한 시각화입니다.
        X축은 시간, Y축은 주파수(Hz), 색상의 밝기는 해당 시간·주파수에서의 에너지(dB)를 나타냅니다.
        밝은 색(노란색)일수록 그 주파수 성분이 강하게 존재함을 의미하며, 성악·음색·화음 구조를 직관적으로 비교할 수 있습니다.
        </div>""", unsafe_allow_html=True)

    with tab2:
        st.plotly_chart(make_bar_fig(r['w1'], r['w2']), use_container_width=True)
        st.markdown("""<div class="tab-desc">
        <b>막대 파형</b>은 오디오 신호를 일정 구간으로 나눠 각 구간의 진폭 절댓값을 막대로 표현한 시각화입니다.
        강하게 부른 부분(포르테)은 막대가 높고, 약하게 부른 부분(피아노)은 낮게 표시됩니다.
        원곡(파랑)과 녹음본(주황)을 겹쳐서 표시하므로, 소리의 강약 패턴이 얼마나 일치하는지 구간별로 한눈에 비교할 수 있습니다.
        </div>""", unsafe_allow_html=True)

    with tab3:
        st.plotly_chart(make_energy_heatmap_fig(r['y1'], r['y2']), use_container_width=True)
        st.markdown("""<div class="tab-desc">
        <b>에너지 히트맵</b>은 RMS(Root Mean Square, 제곱평균제곱근) 에너지를 색상 그라디언트로 표현한 시각화입니다.
        RMS는 특정 구간 내 신호 에너지의 평균적 크기를 나타내는 지표로, 소리의 실제 음량감과 밀접하게 관련됩니다.
        막대의 색상이 진할수록(원곡: 진한 파랑, 녹음본: 진한 주황) 해당 구간의 에너지가 높음을 의미하며, 발성의 힘과 지속성을 분석하는 데 유용합니다.
        </div>""", unsafe_allow_html=True)

    with tab4:
        st.plotly_chart(make_line_fig(r['w1'], r['w2']), use_container_width=True)
        st.markdown("""<div class="tab-desc">
        <b>라인 파형</b>은 원시 오디오 신호의 진폭을 시간축 위에 직접 표현한 기본 파형입니다.
        원곡(실선, 파랑)과 녹음본(점선, 주황)을 겹쳐서 표시해 파형의 세밀한 모양 차이를 비교합니다.
        두 파형의 피크(최고점)와 밸리(최저점)의 위치·크기가 얼마나 일치하는지, 전체적인 파형 패턴을 시각적으로 확인할 수 있습니다.
        </div>""", unsafe_allow_html=True)

    st.markdown("### 유사도 분석")
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.markdown(f"""
        <div class="score-box">
          <div class="score-label">전체 유사도</div>
          <div class="score-value">{r['overall']:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with col_b:
        st.markdown(f"""
        <div class="score-box">
          <div class="score-label">진폭 일치도</div>
          <div class="score-value">{r['rms_score']:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with col_c:
        st.markdown(f"""
        <div class="score-box">
          <div class="score-label">음정 일치도</div>
          <div class="score-value">{r['p_score']:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with col_d:
        st.markdown(f"""
        <div class="score-box">
          <div class="score-label">음색 일치도</div>
          <div class="score-value">{r['mfcc_score']:.1f}%</div>
        </div>""", unsafe_allow_html=True)

    st.progress(int(r['overall']))

    st.markdown("### 결과 다운로드")
    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        pdf_bytes = generate_pdf(r)
        st.download_button(
            label="PDF 다운로드",
            data=pdf_bytes,
            file_name="분석결과.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    with dl_col2:
        xlsx_bytes = generate_xlsx(r)
        st.download_button(
            label="XLSX 다운로드",
            data=xlsx_bytes,
            file_name="분석결과.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

# ── Footer ────────────────────────────────────────────────
st.markdown(
    '<div class="custom-footer">© 2026 이대형 · '
    '<a href="https://www.instagram.com/aicreator.z/" target="_blank" '
    'style="color:#94A3B8; text-decoration:none;">@aicreator.z</a></div>',
    unsafe_allow_html=True,
)
