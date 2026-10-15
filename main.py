
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# -----------------------------
# 기본 설정
# -----------------------------
st.set_page_config(
    page_title="서울 기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)
기준연도 = 2025
최소관측일수 = 300

st.title("🌡️ 서울 기온 예측기")
st.write(
    "서울의 연평균기온을 이용해 회귀 직선을 만들고, "
    "선택한 연도의 예상 평균기온을 확인합니다."
)


# -----------------------------
# 데이터 불러오기
# -----------------------------
@st.cache_data
def 데이터_불러오기():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    필요한열 = ["날짜", "지점", "평균기온", "최저기온", "최고기온"]
    없는열 = [열 for 열 in 필요한열 if 열 not in df.columns]

    if 없는열:
        raise ValueError(
            f"CSV에 필요한 열이 없습니다: {', '.join(없는열)}"
        )

    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    df["평균기온"] = pd.to_numeric(
        df["평균기온"], errors="coerce"
    )

    df = df.dropna(subset=["날짜", "평균기온"]).copy()

    # 기준 기간은 2025년까지
    df = df[df["날짜"].dt.year <= 기준연도].copy()

    # 연도별 평균기온과 실제 관측일 수 계산
    연도별 = (
        df.groupby(df["날짜"].dt.year)
        .agg(
            연평균기온=("평균기온", "mean"),
            관측일수=("날짜", "nunique")
        )
        .reset_index()
        .rename(columns={"날짜": "연도"})
    )

    # 관측일이 300일 미만인 해 제외
    연도별 = 연도별[
        연도별["관측일수"] >= 최소관측일수
    ].copy()

    연도별["연도"] = 연도별["연도"].astype(int)
    연도별 = 연도별.sort_values("연도").reset_index(drop=True)

    return 연도별


# -----------------------------
# 데이터 처리 및 회귀 분석
# -----------------------------
try:
    연도별 = 데이터_불러오기()
except Exception as e:
    st.error(f"데이터를 불러오거나 처리하지 못했습니다: {e}")
    st.info("인터넷 연결과 CSV 주소를 확인해 주세요.")
    st.stop()

if len(연도별) < 2:
    st.error("회귀 분석을 수행하기에 유효한 연도가 부족합니다.")
    st.stop()

연도 = 연도별["연도"].to_numpy(dtype=float)
기온 = 연도별["연평균기온"].to_numpy(dtype=float)

# 최소제곱법으로 회귀 직선 계산
기울기, 절편 = np.polyfit(연도, 기온, 1)

# 상관계수
상관계수 = np.corrcoef(연도, 기온)[0, 1]

시작연도 = int(연도별["연도"].min())
끝연도 = int(연도별["연도"].max())
사용연도수 = len(연도별)


# -----------------------------
# 연도 선택 슬라이더
# -----------------------------
st.subheader("예측할 연도 선택")

선택연도 = st.slider(
    "연도",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1,
    format="%d년"
)

예상기온 = 기울기 * 선택연도 + 절편

# -----------------------------
# 예상 기온 크게 표시
# -----------------------------
st.markdown(f"### {선택연도}년 서울 예상 연평균기온")

st.markdown(
    f"""
    <div style="
        background-color: #eef6ff;
        border-radius: 16px;
        padding: 28px 16px;
        text-align: center;
        margin: 10px 0 20px 0;
    ">
        <div style="
            color: #245a91;
            font-size: 20px;
            margin-bottom: 8px;
        ">회귀 직선에 따른 예상 기온</div>
        <div style="
            color: #174a7e;
            font-size: 56px;
            font-weight: bold;
        ">{예상기온:.2f} °C</div>
    </div>
    """,
    unsafe_allow_html=True
)

st.caption(
    "예측값은 과거 연평균기온에 맞춘 직선을 연장한 추정치입니다. "
    "실제 미래 기온이나 기후 예측 모델의 결과와는 다를 수 있습니다."
)


# -----------------------------
# 핵심 분석 지표
# -----------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("상관계수", f"{상관계수:.4f}")

with col2:
    st.metric("회귀에 사용한 연도 수", f"{사용연도수:,}년")

with col3:
    st.metric(
        "회귀 분석 기간",
        f"{시작연도}년 ~ {끝연도}년"
    )

# -----------------------------
# 회귀 직선 정보
# -----------------------------
st.subheader("회귀 직선 정보")

st.write(
    f"**회귀식:** 연평균기온 = "
    f"{기울기:.6f} × 연도 + ({절편:.4f})"
)

st.write(
    f"**100년간 기온 변화량:** "
    f"{기울기 * 100:+.2f} °C"
)

st.write(
    f"회귀 직선은 {시작연도}년부터 {끝연도}년까지의 "
    f"{사용연도수}개 연평균기온 자료를 사용해 만들었습니다."
)

# -----------------------------
# Plotly 산점도 + 회귀 직선
# -----------------------------
st.subheader("연도별 평균기온과 회귀 직선")

fig = go.Figure()

# 실제 관측값 산점도
fig.add_trace(
    go.Scatter(
        x=연도별["연도"],
        y=연도별["연평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(size=7, opacity=0.75),
        customdata=연도별[["관측일수"]].to_numpy(),
        hovertemplate=(
            "%{x}년<br>"
            "연평균기온: %{y:.2f} °C<br>"
            "관측일수: %{customdata[0]}일"
            "<extra></extra>"
        )
    )
)

# 관측 기간에 맞춘 회귀 직선
회귀연도 = np.linspace(시작연도, 끝연도, 200)
회귀기온 = 기울기 * 회귀연도 + 절편

fig.add_trace(
    go.Scatter(
        x=회귀연도,
        y=회귀기온,
        mode="lines",
        name="회귀 직선",
        line=dict(width=3)
    )
)

# 선택한 연도의 예상값을 별도 표시
fig.add_trace(
    go.Scatter(
        x=[선택연도],
        y=[예상기온],
        mode="markers",
        name=f"{선택연도}년 예상값",
        marker=dict(size=14, symbol="diamond"),
        hovertemplate=(
            "%{x}년 예상기온: %{y:.2f} °C"
            "<extra></extra>"
        )
    )
)

fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    hovermode="closest",
    template="plotly_white",
    height=550,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0
    ),
    margin=dict(l=20, r=20, t=45, b=20)
)

fig.update_xaxes(
    dtick=10,
    showgrid=True
)
fig.update_yaxes(showgrid=True)

st.plotly_chart(fig, width="stretch")

# -----------------------------
# 연도별 데이터 확인
# -----------------------------
with st.expander("회귀 분석에 사용한 연도별 데이터 보기"):
    표시용 = 연도별.copy()
    표시용["연평균기온"] = 표시용["연평균기온"].round(2)
    st.dataframe(표시용, width="stretch", hide_index=True)

# -----------------------------
# 데이터 기준 안내
# -----------------------------
st.caption(
    "데이터 출처: 기상청 서울 기온 CSV | "
    "분석 기준: 2025년까지, 연간 관측일 300일 이상"
)
