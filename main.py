
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

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
st.write("서울의 연평균기온 자료로 장기 기온 변화 추세를 분석합니다.")


@st.cache_data
def 데이터_불러오기():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    필요한열 = ["날짜", "지점", "평균기온", "최저기온", "최고기온"]
    없는열 = [열 for 열 in 필요한열 if 열 not in df.columns]

    if 없는열:
        raise ValueError(f"필요한 열이 없습니다: {없는열}")

    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")
    df = df.dropna(subset=["날짜", "평균기온"]).copy()

    # 2025년까지의 자료만 사용
    df = df[df["날짜"].dt.year <= 기준연도].copy()

    # 연도별 평균기온과 관측일 수
    연도별 = (
        df.assign(연도=df["날짜"].dt.year)
        .groupby("연도")
        .agg(
            연평균기온=("평균기온", "mean"),
            관측일수=("날짜", "nunique")
        )
        .reset_index()
    )

    # 관측일이 300일 미만인 연도 제외
    연도별 = 연도별[
        연도별["관측일수"] >= 최소관측일수
    ].copy()

    연도별["연도"] = 연도별["연도"].astype(int)

    return 연도별.sort_values("연도").reset_index(drop=True)


def 회귀분석(df):
    """연평균기온에 대한 선형회귀 결과를 계산합니다."""
    if len(df) < 2:
        return None

    x = df["연도"].to_numpy(dtype=float)
    y = df["연평균기온"].to_numpy(dtype=float)

    기울기, 절편 = np.polyfit(x, y, 1)
    상관계수 = np.corrcoef(x, y)[0, 1]

    return {
        "기울기": 기울기,
        "절편": 절편,
        "100년변화량": 기울기 * 100,
        "상관계수": 상관계수,
        "개수": len(df),
        "시작연도": int(df["연도"].min()),
        "끝연도": int(df["연도"].max()),
    }


try:
    연도별 = 데이터_불러오기()
except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()

if len(연도별) < 2:
    st.error("회귀 분석을 수행할 자료가 부족합니다.")
    st.stop()

# 전체 기간 회귀
전체결과 = 회귀분석(연도별)

# 최근 20년: 2025년을 기준으로 2006~2025년
최근20년시작 = 기준연도 - 19
최근20년자료 = 연도별[
    (연도별["연도"] >= 최근20년시작)
    & (연도별["연도"] <= 기준연도)
].copy()

최근결과 = 회귀분석(최근20년자료)

# -----------------------------
# 기울기 비교: 100년당 몇 도?
# -----------------------------
st.subheader("📈 100년당 기온 변화량")

전체열, 최근열 = st.columns(2)

with 전체열:
    st.markdown("#### 전체 관측 기간")
    st.metric(
        label="100년당 기온 변화량",
        value=f"{전체결과['100년변화량']:+.2f} °C",
        help="전체 유효 연도의 회귀 직선 기울기를 100배 한 값입니다."
    )
    st.caption(
        f"분석 기간: {전체결과['시작연도']}~"
        f"{전체결과['끝연도']}년"
    )
    st.caption(f"회귀에 사용한 연도 수: {전체결과['개수']}년")

with 최근열:
    st.markdown("#### 최근 20년")
    if 최근결과 is not None:
        st.metric(
            label="100년당 기온 변화량",
            value=f"{최근결과['100년변화량']:+.2f} °C",
            help=(
                f"{최근20년시작}~{기준연도}년 범위 안에서 "
                "관측일 조건을 통과한 연도만 사용합니다."
            )
        )
        st.caption(
            f"분석 기간: {최근결과['시작연도']}~"
            f"{최근결과['끝연도']}년"
        )
        st.caption(f"회귀에 사용한 연도 수: {최근결과['개수']}년")
    else:
        st.warning("최근 20년 자료가 부족해 회귀 직선을 만들 수 없습니다.")

st.caption(
    "양수(+)는 기온 상승 추세, 음수(-)는 기온 하락 추세를 뜻합니다. "
    "100년 변화량은 회귀 직선의 기울기를 100배 한 값입니다."
)

st.divider()

# -----------------------------
# 연도 슬라이더 및 예상 기온
# -----------------------------
st.subheader("🔮 연도별 예상 평균기온")

선택연도 = st.slider(
    "예측할 연도",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1,
    format="%d년"
)

예상기온 = (
    전체결과["기울기"] * 선택연도
    + 전체결과["절편"]
)

st.markdown(
    f"""
    <div style="
        background-color: #eaf3ff;
        border-radius: 18px;
        padding: 25px;
        text-align: center;
        margin: 10px 0 20px 0;
    ">
        <div style="font-size: 22px; color: #245a91;">
            {선택연도}년 예상 연평균기온
        </div>
        <div style="
            font-size: 58px;
            font-weight: bold;
            color: #174a7e;
            padding-top: 8px;
        ">
            {예상기온:.2f} °C
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.caption(
    "예상 기온은 전체 관측 기간의 회귀 직선을 연장한 값입니다. "
    "실제 미래 기온을 보장하지 않습니다."
)

# -----------------------------
# 산점도와 회귀 직선
# -----------------------------
st.subheader("연도별 평균기온과 회귀 직선")

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=연도별["연도"],
        y=연도별["연평균기온"],
        mode="markers",
        name="연평균기온",
        marker=dict(size=7, opacity=0.7),
        customdata=연도별[["관측일수"]].to_numpy(),
        hovertemplate=(
            "%{x}년<br>"
            "평균기온: %{y:.2f} °C<br>"
            "관측일: %{customdata[0]}일"
            "<extra></extra>"
        )
    )
)

# 전체 기간 회귀 직선
회귀연도 = np.linspace(
    전체결과["시작연도"],
    전체결과["끝연도"],
    200
)
회귀기온 = (
    전체결과["기울기"] * 회귀연도
    + 전체결과["절편"]
)

fig.add_trace(
    go.Scatter(
        x=회귀연도,
        y=회귀기온,
        mode="lines",
        name="전체 기간 회귀 직선",
        line=dict(width=3)
    )
)

# 최근 20년 회귀 직선
if 최근결과 is not None:
    최근회귀연도 = np.linspace(
        최근결과["시작연도"],
        최근결과["끝연도"],
        100
    )
    최근회귀기온 = (
        최근결과["기울기"] * 최근회귀연도
        + 최근결과["절편"]
    )

    fig.add_trace(
        go.Scatter(
            x=최근회귀연도,
            y=최근회귀기온,
            mode="lines",
            name="최근 20년 회귀 직선",
            line=dict(width=3, dash="dash")
        )
    )

# 선택 연도의 전체 기간 회귀 예측값
fig.add_trace(
    go.Scatter(
        x=[선택연도],
        y=[예상기온],
        mode="markers",
        name=f"{선택연도}년 예상 기온",
        marker=dict(size=14, symbol="diamond"),
        hovertemplate="%{x}년: %{y:.2f} °C<extra></extra>"
    )
)

fig.update_layout(
    template="plotly_white",
    height=560,
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    hovermode="closest",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0
    ),
    margin=dict(l=20, r=20, t=50, b=20)
)

fig.update_xaxes(dtick=10, showgrid=True)
fig.update_yaxes(showgrid=True)

st.plotly_chart(fig, width="stretch")

# -----------------------------
# 상관계수와 회귀 분석 정보
# -----------------------------
st.subheader("📊 회귀 분석 정보")

정보1, 정보2 = st.columns(2)

with 정보1:
    st.markdown("#### 전체 기간")
    st.write(f"- 상관계수: **{전체결과['상관계수']:.4f}**")
    st.write(f"- 회귀식: `기온 = {전체결과['기울기']:.6f} × 연도 + ({전체결과['절편']:.4f})`")
    st.write(f"- 사용 연도 수: **{전체결과['개수']}년**")
    st.write(f"- 시작 연도: **{전체결과['시작연도']}년**")
    st.write(f"- 끝 연도: **{전체결과['끝연도']}년**")

with 정보2:
    st.markdown("#### 최근 20년")
    if 최근결과 is not None:
        st.write(f"- 상관계수: **{최근결과['상관계수']:.4f}**")
        st.write(f"- 회귀식: `기온 = {최근결과['기울기']:.6f} × 연도 + ({최근결과['절편']:.4f})`")
        st.write(f"- 사용 연도 수: **{최근결과['개수']}년**")
        st.write(f"- 시작 연도: **{최근결과['시작연도']}년**")
        st.write(f"- 끝 연도: **{최근결과['끝연도']}년**")
    else:
        st.write("회귀 분석에 필요한 연도가 부족합니다.")

with st.expander("회귀 분석에 사용한 연도별 데이터"):
    표시자료 = 연도별.copy()
    표시자료["연평균기온"] = 표시자료["연평균기온"].round(2)
    st.dataframe(표시자료, width="stretch", hide_index=True)

st.caption(
    "데이터: 서울 기온 CSV | 기준 연도: 2025년 | "
    "연간 관측일 300일 이상인 연도만 사용"
)
