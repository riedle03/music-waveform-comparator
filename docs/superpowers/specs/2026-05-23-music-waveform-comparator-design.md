# 노래 파형 비교기 — 설계 문서

**날짜**: 2026-05-23  
**작성자**: 이대형 (riedel@e-mirim.hs.kr)  
**상태**: 승인됨

---

## 1. 개요

학생이 원곡 MP3와 자신이 녹음한 MP3를 업로드하면, 두 파일의 파형을 분석해 유사도를 시각적으로 비교하는 웹 앱. 학교 음악 수업 과제 제출 및 개인 연습 용도로 사용.

**핵심 제약**
- 로그인 없음 — 누구나 바로 사용
- 서버 직접 소유 불필요 — Streamlit Cloud 무료 배포
- 유료 API 없음 — 모든 분석은 librosa (오픈소스)

---

## 2. 기술 스택

| 역할 | 기술 | 비고 |
|------|------|------|
| 앱 프레임워크 | Streamlit | Python 단일 파일로 UI + 분석 |
| 오디오 분석 | librosa + ffmpeg | MP3 직접 로드 지원 |
| 파형 시각화 | Plotly | 인터랙티브 (확대/축소/구간 선택) |
| 폰트 | Pretendard | CSS 주입으로 적용 |
| 배포 | Streamlit Cloud | GitHub 연결, 무료 |

---

## 3. 프로젝트 구조

```
음악프로젝트/
├── app.py                  ← 메인 Streamlit 앱
├── analyzer.py             ← librosa 분석 로직
├── requirements.txt        ← streamlit, librosa, plotly, numpy
└── .streamlit/
    └── config.toml         ← 테마 설정
```

---

## 4. 화면 구성

### 4.1 레이아웃

```
┌─────────────────────────────────────────┐
│  [아이콘] 노래 파형 비교기              │  ← Pretendard Bold, 큰 타이틀
│  원곡과 녹음본의 파형을 분석합니다      │  ← 부제 (회색)
├─────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐    │
│  │  원곡 MP3    │  │  녹음본 MP3  │    │  ← 2열 업로드
│  │  [파일 선택] │  │  [파일 선택] │    │
│  └──────────────┘  └──────────────┘    │
├─────────────────────────────────────────┤
│  [채움 파형] [막대 파형] [라인 파형]   │  ← st.tabs() 3개 (탭 내부에 Plotly 차트)
│  (두 파일 업로드 시 자동 분석)         │
│  (분석 중: st.spinner + st.progress)   │
├─────────────────────────────────────────┤
│  전체 유사도              73%           │
│  진폭 일치도   80%                     │
│  음정 일치도   68%   (2단계, 준비 중)  │
│  음색 일치도   72%   (3단계, 준비 중)  │
├─────────────────────────────────────────┤
│  © 2026 이대형 · riedel@e-mirim.hs.kr  │
└─────────────────────────────────────────┘
```

### 4.2 파형 탭 3종

| 탭 | 시각 스타일 | Plotly 구현 | 용도 |
|----|------------|-------------|------|
| 채움 파형 | 위아래 대칭, 반투명 채움 | `go.Scatter(fill='tozeroy')` | 전체 에너지 패턴 |
| 막대 파형 | 블록 막대 | `go.Bar()` | 구간별 강약 비교 |
| 라인 파형 | 얇은 선 (원곡 실선, 녹음 점선) | `go.Scatter()` | 세밀한 파형 차이 |

- 원곡: 파란색 / 녹음본: 주황색 (모든 탭 공통)
- 두 트랙 겹쳐서 표시 (overlay)

---

## 5. 분석 로직 — 단계별 로드맵

### 1단계 (MVP): 진폭(RMS) 비교

```python
@st.cache_data
def analyze_rms(file1_bytes, file2_bytes):
    y1, sr = librosa.load(file1_bytes)
    y2, sr = librosa.load(file2_bytes)

    rms1 = librosa.feature.rms(y=y1)[0]
    rms2 = librosa.feature.rms(y=y2)[0]

    min_len = min(len(rms1), len(rms2))
    score = 1 - np.mean(np.abs(rms1[:min_len] - rms2[:min_len]))
    return float(score) * 100
```

- 처리 시간: ~0.5초
- `@st.cache_data`: 같은 파일 재업로드 시 즉시 반환

### 2단계: 피치(음높이) 비교

```python
pitch1 = librosa.yin(y1, fmin=80, fmax=1000)
pitch2 = librosa.yin(y2, fmin=80, fmax=1000)

# 무음 구간 제거 후 비교
voiced1 = pitch1[pitch1 > 0]
voiced2 = pitch2[pitch2 > 0]
```

- 처리 시간: ~2-4초
- 음정 일치 여부 판단 (노래 비교에서 가장 중요한 지표)

### 3단계: MFCC + DTW (음색 + 박자 보정)

```python
mfcc1 = librosa.feature.mfcc(y=y1, sr=sr, n_mfcc=13)
mfcc2 = librosa.feature.mfcc(y=y2, sr=sr, n_mfcc=13)

D, wp = librosa.sequence.dtw(mfcc1, mfcc2)
dtw_score = 1 / (1 + D[-1, -1] / len(wp))
```

- 처리 시간: ~3-5초
- DTW: 학생이 빠르게/느리게 불러도 시간축 자동 정렬 후 비교

### 유사도 종합 가중치

| 항목 | 가중치 | 이유 |
|------|--------|------|
| RMS 진폭 | 20% | 강약 패턴 |
| 피치 | 40% | 음정 (노래의 핵심) |
| MFCC + DTW | 40% | 음색 + 박자 보정 |

**전체 처리 시간**: 3단계 합산 10초 이내 (3분짜리 MP3 기준)

---

## 6. 스타일 가이드

- **폰트**: Pretendard (CSS 주입, Google Fonts CDN)
- **아이콘**: SVG 또는 HTML 유니코드 (react-icons 사용 불가 — Streamlit 제약)
- **색상**: 원곡 `#3B82F6` (파랑), 녹음본 `#F97316` (주황)
- **Footer**: `© 2026 이대형 · riedel@e-mirim.hs.kr` — `st.markdown()` HTML 주입

---

## 7. 배포

1. GitHub 저장소 생성 후 코드 푸시
2. [share.streamlit.io](https://share.streamlit.io) 접속 → GitHub 연결
3. `app.py` 지정 → 자동 빌드 및 배포
4. 무료 URL 발급 (예: `username-music-compare.streamlit.app`)

**ffmpeg 설치**: `packages.txt` 파일에 `ffmpeg` 한 줄 추가로 Streamlit Cloud에서 자동 설치됨.

---

## 8. 미결 사항 및 향후 과제

- 2단계, 3단계 분석 UI — 1단계 완료 후 순차 추가
- 파일 크기 제한 — Streamlit Cloud 기본 200MB, MP3는 충분
- 음악 저작권 — 파일이 서버에 저장되지 않고 분석 후 즉시 폐기
