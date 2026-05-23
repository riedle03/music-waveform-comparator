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
