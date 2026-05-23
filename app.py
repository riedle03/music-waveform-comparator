import streamlit as st
import numpy as np
import plotly.graph_objects as go
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

# ── Footer ────────────────────────────────────────────────
st.markdown(
    '<div class="custom-footer">© 2026 이대형 · riedel@e-mirim.hs.kr</div>',
    unsafe_allow_html=True,
)
