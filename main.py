# main.py
import io
import os
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# ---------- 한글 폰트 & 스타일 ----------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
   font-family: 'Noto Sans KR', 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif;
}
</style>
""",
    unsafe_allow_html=True,
)


# ---------- CSV 읽기 (인코딩 대응) ----------
@st.cache_data
def read_csv(file):
    """utf-8-sig → cp949 순서로 시도, bytes로 한 번만 읽음"""
    data = file.getvalue()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()


# ---------- 열 이름 표준화 ----------
def std_cols(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        k = c.lower().replace(" ", "")
        if k in ("time", "categories", "시간", "날짜", "datetime", "date"):
            rename[c] = "time"
        elif k in ("temperature", "온도"):
            rename[c] = "temperature"
        elif k in ("humidity", "습도"):
            rename[c] = "humidity"
        elif k == "ph":
            rename[c] = "ph"
        elif k == "ec":
            rename[c] = "ec"
        elif "생중량" in k:
            rename[c] = "생중량"
        elif "잎" in k:
            rename[c] = "잎 수"
        elif "지상부" in k and "길이" in k:
            rename[c] = "지상부 길이"
        elif "지하부" in k and "길이" in k:
            rename[c] = "지하부 길이"
        elif "지상부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지상부 질량"
        elif "지하부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지하부 질량"
        elif "개체" in k:
            rename[c] = "개체번호"
    df = df.rename(columns=rename)
    # 숫자 열 강제 변환
    for c in df.columns:
        if c in ("time", "개체번호"):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            num = pd.to_numeric(
                df[c].astype(str).str.replace(",", "")
                .str.extract(r"(-?\d+\.?\d*)")[0],
                errors="coerce",
            )
            if num.notna().mean() >= 0.5:
                df[c] = num
    # 생중량 없으면 지상부+지하부 질량 합으로 계산
    if "생중량" not in df.columns and {"지상부 질량", "지하부 질량"} <= set(df.columns):
        df["생중량"] = df["지상부 질량"] + df["지하부 질량"]
    # 시간 파싱
    if "time" in df.columns:
        t = df["time"].astype(str).str.strip()
        parsed = pd.to_datetime(t, errors="coerce", format="mixed")
        bad = parsed.isna() & t.str.match(r"^\d{8}_\d{1,2}$")
        if bad.any():
            parts = t[bad].str.split("_", expand=True)
            parsed[bad] = pd.to_datetime(parts[0], format="%Y%m%d") + pd.to_timedelta(
                parts[1].astype(int), unit="h"
            )
        df["time"] = parsed
    return df


# ---------- 학교명 추출 ----------
def extract_school(filename):
    base = os.path.splitext(os.path.basename(filename))[0]
    cleaned = re.sub(r"[\d\-\._]+", "", base).strip()
    return cleaned if cleaned else "우리 학교"


# ---------- Plotly 공통 폰트 적용 헬퍼 ----------
def apply_font(fig):
    fig.update_layout(
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


# ---------- 막대 그래프 (일원화) ----------
def bar_chart(df, value_col, title, color="#4CAF50", highlight_vals=None):
    """개체번호 제외한 막대 그래프. highlight_vals에 해당하면 강조색."""
    if "개체번호" in df.columns:
        d = df[["개체번호", value_col]].dropna().copy()
        labels = d["개체번호"].astype(str).tolist()
        vals = d[value_col].tolist()
    else:
        d = df[[value_col]].dropna().copy()
        labels = [f"#{i+1}" for i in range(len(d))]
        vals = d[value_col].tolist()
    colors = [
        "#FF5722" if (highlight_vals is not None and v in highlight_vals) else color
        for v in vals
    ]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels,
        y=vals,
        marker_color=colors,
        hovertemplate="%{x}<br>%{y:.2f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=14)),
        xaxis_title="개체",
        yaxis_title=value_col,
        showlegend=False,
    )
    return apply_font(fig)


# ============================================================
#  사이드바
# ============================================================
st.sidebar.header("📁 파일 업로드")
env_file = st.sidebar.file_uploader(
    "환경 데이터 CSV", type="csv",
    help="utf-8-sig 또는 cp949 인코딩 지원",
)
growth_file = st.sidebar.file_uploader(
    "생육 결과 데이터 CSV", type="csv",
    help="utf-8-sig 또는 cp949 인코딩 지원",
)

st.sidebar.markdown("---")
st.sidebar.subheader("📏 생육 측정 항목")
growth_cols = st.sidebar.multiselect(
    "비교할 항목 선택",
    ["잎 수", "지상부 길이", "지하부 길이", "생중량"],
    default=["잎 수", "지상부 길이", "지하부 길이", "생중량"],
)

# ============================================================
#  데이터 로딩 (한 번만)
# ============================================================
env_df = pd.DataFrame()
growth_df = pd.DataFrame()
school_name = "우리 학교"

if env_file is not None:
    with st.spinner("환경 데이터를 읽는 중..."):
        env_df = read_csv(env_file)
        env_df = std_cols(env_df)
        school_name = extract_school(env_file.name)

if growth_file is not None:
    with st.spinner("생육 데이터를 읽는 중..."):
        growth_df = read_csv(growth_file)
        growth_df = std_cols(growth_df)
        if not school_name or school_name == "우리 학교":
            school_name = extract_school(growth_file.name)


# ============================================================
#  제목
# ============================================================
st.set_page_config(page_title="극지식물 생육 대시보드", layout="wide")
st.title("🌱 우리 학교 극지식물 개체별 생육 차이 분석")
if school_name and school_name != "우리 학교":
    st.caption(f"📍 {school_name} 극지식물 실험 데이터")


# ============================================================
#  Tab 1 — 실험 개요
# ============================================================
tab1, tab2, tab3 = st.tabs(["📖 실험 개요", "🌡️ 환경 데이터", "📊 생육 결과"])

with tab1:
    try:
        st.header("📖 실험 개요")

        st.subheader("🔬 연구 배경 및 목적")
        st.markdown(
            f"""
**극지식물(Extremophyte)** 은 건조, 고염, 극한 온도 등 척박한 환경에서 생존하는 식물입니다.  
이 대시보드는 **{school_name}** 에서 재배한 극지식물 개체들의 생육 차이를 분석하여,

- 개체마다 생중량(전체 무게)에 얼마나 큰 차이가 있는지  
- 잎 수·지상부 길이와 생중량이 어떤 관계인지  
- 잘 자란 개체(상위 25%)와 덜 자란 개체(하위 25%)는 어떤 차이가 있는지  

를 시각적으로 탐색합니다.

> **통계 용어 짧은 설명**
> - **평균(Mean)**: 모든 값을 더해 개수로 나눈 대표값
> - **표준편차(Std)**: 값이 평균에서 얼마나 넓게 퍼져 있는지 나타내는 지표
> - **변동계수(CV)**: 표준편차를 평균으로 나눈 값(%) — 서로 다른 규모의 데이터를 비교할 때 유용
> - **상관계수(r)**: 두 변수가 함께 움직이는 정도를 -1 ~ +1 사이 숫자로 표현. +1에 가까울수록 함께 증가
> """
        )

        if env_df.empty and growth_df.empty:
            st.info("📁 사이드바에서 CSV 파일을 업로드하면 분석 결과가 표시됩니다.")
        else:
            st.subheader("📋 데이터 요약")
            summary_rows = []
            if not env_df.empty:
                time_col = "time" if "time" in env_df.columns else None
                if time_col and pd.api.types.is_datetime64_any_dtype(env_df[time_col]):
                    tmin = env_df[time_col].min().strftime("%Y-%m-%d")
                    tmax = env_df[time_col].max().strftime("%Y-%m-%d")
                elif time_col:
                    tmin = str(env_df[time_col].min())
                    tmax = str(env_df[time_col].max())
                else:
                    tmin, tmax = "알 수 없음", "알 수 없음"
                summary_rows.append({
                    "구분": "환경 데이터",
                    "개체(측정 횟수)": len(env_df),
                    "측정 기간": f"{tmin} ~ {tmax}",
                    "측정 항목": "온도, 습도, pH, EC",
                })
            if not growth_df.empty:
                present = [c for c in ["잎 수", "지상부 길이", "지하부 길이", "생중량"] if c in growth_df.columns]
                summary_rows.append({
                    "구분": "생육 결과 데이터",
                    "개체(측정 횟수)": len(growth_df),
                    "측정 기간": "실험 종료일 기준 1회 측정",
                    "측정 항목": ", ".join(present),
                })
            if summary_rows:
                st.dataframe(pd.DataFrame(summary_rows), width="stretch")

            st.subheader("📊 주요 지표")
            c1, c2, c3, c4 = st.columns(4)

            total = len(growth_df) if not growth_df.empty else 0
            c1.metric("총 개체 수", total if total else "—")

            if "temperature" in env_df.columns and not env_df.empty:
                avg_temp = env_df["temperature"].dropna().mean()
                c2.metric("평균 온도", f"{avg_temp:.1f}°C")
            else:
                c2.metric("평균 온도", "—")

            if "humidity" in env_df.columns and not env_df.empty:
                avg_humid = env_df["humidity"].dropna().mean()
                c3.metric("평균 습도", f"{avg_humid:.1f}%")
            else:
                c3.metric("평균 습도", "—")

            if not growth_df.empty and "생중량" in growth_df.columns:
                avg_bio = growth_df["생중량"].dropna().mean()
                c4.metric("평균 생중량", f"{avg_bio:.2f} g")
            else:
                c4.metric("평균 생중량", "—")

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")


# ============================================================
#  Tab 2 — 환경 데이터
# ============================================================
with tab2:
    try:
        if env_df.empty:
            st.info(
                "📁 사이드바에서 환경 데이터 CSV를 업로드하면 환경 분석 결과가 표시됩니다."
            )
        else:
            st.header("🌡️ 환경 데이터")

            st.subheader("📈 환경 요약 통계")
            env_num_cols = [
                c for c in ["temperature", "humidity", "ph", "ec"]
                if c in env_df.columns
            ]
            if env_num_cols:
                env_clean = env_df[env_num_cols].dropna()
                if len(env_clean) >= 2:
                    sum_tbl = pd.DataFrame({
                        "측정 항목": [
                            "온도 (°C)" if c == "temperature" else
                            "습도 (%)" if c == "humidity" else
                            "pH" if c == "ph" else
                            "EC (mS/cm)"
                            for c in env_num_cols
                        ],
                        "평균": [round(env_clean[c].mean(), 2) for c in env_num_cols],
                        "최소": [round(env_clean[c].min(), 2) for c in env_num_cols],
                        "최대": [round(env_clean[c].max(), 2) for c in env_num_cols],
                    })
                    st.dataframe(sum_tbl, width="stretch")
                else:
                    st.info("🌡️ 환경 데이터가 충분하지 않아 요약 통계를 표시할 수 없습니다.")
            else:
                st.info("🌡️ 온도·습도·pH·EC 열이 없어 환경 요약을 표시할 수 없습니다.")

            st.subheader("📅 날짜별 환경 변화 (일별 평균)")
            time_col = "time" if "time" in env_df.columns else None
            if time_col and pd.api.types.is_datetime64_any_dtype(env_df[time_col]):
                env_ts = env_df.copy()
                env_ts["date"] = env_ts[time_col].dt.date
                daily = (
                    env_ts.groupby("date")[env_num_cols].mean().reset_index()
                    if env_num_cols else pd.DataFrame()
                )

                if not daily.empty:
                    ts_cols = [
                        c for c in ["temperature", "humidity", "ph", "ec"]
                        if c in daily.columns
                    ]
                    if ts_cols:
                        n_ts = len(ts_cols)
                        ts_fig = make_subplots(
                            rows=n_ts, cols=1,
                            subplot_titles=[
                                "온도 변화 (°C)" if c == "temperature" else
                                "습도 변화 (%)" if c == "humidity" else
                                "pH 변화" if c == "ph" else
                                "EC 변화 (mS/cm)"
                                for c in ts_cols
                            ],
                            shared_xaxes=True,
                        )
                        colors_ts = {
                            "temperature": "#FF5722",
                            "humidity": "#2196F3",
                            "ph": "#4CAF50",
                            "ec": "#9C27B0",
                        }
                        for i, c in enumerate(ts_cols, 1):
                            ts_fig.add_trace(
                                go.Scatter(
                                    x=daily["date"],
                                    y=daily[c],
                                    mode="lines",
                                    name=c,
                                    line=dict(
                                        color=colors_ts.get(c, "#607D8B"),
                                        width=2,
                                    ),
                                    hovertemplate="%{x}<br>%{y:.2f}<extra></extra>",
                                ),
                                row=i, col=1,
                            )
                            ts_fig.update_yaxes(title_text="", row=i, col=1)
                        ts_fig.update_layout(
                            height=200 * n_ts,
                            showlegend=False,
                            xaxis_title="날짜",
                        )
                        apply_font(ts_fig)
                        st.plotly_chart(ts_fig, width="stretch")
                    else:
                        st.info("🌡️ 표시할 환경 열(온도·습도·pH·EC)이 없습니다.")
                else:
                    st.info("🌡️ 날짜별로 집계할 환경 데이터가 충분하지 않습니다.")
            else:
                st.info(
                    "🌡️ 시간 열이 없거나 날짜 형식으로 인식되지 않아 "
                    "시계열 그래프를 표시할 수 없습니다."
                )

            with st.expander("📄 환경 데이터 원본 테이블"):
                st.dataframe(env_df, width="stretch")
                csv_env = env_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="📥 환경 데이터 CSV 다운로드",
                    data=csv_env,
                    file_name=f"{school_name}_환경데이터.csv",
                    mime="text/csv",
                )
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")


# ============================================================
#  Tab 3 — 생육 결과
# ============================================================
with tab3:
    try:
        if growth_df.empty:
            st.info(
                "📁 사이드바에서 생육 결과 데이터 CSV를 업로드하면 "
                "생육 분석 결과가 표시됩니다."
            )
        else:
            st.header("📊 생육 결과")

            bio_col = "생중량" if "생중량" in growth_df.columns else None
            leaf_col = "잎 수" if "잎 수" in growth_df.columns else None
            shoot_col = "지상부 길이" if "지상부 길이" in growth_df.columns else None
            root_col = "지하부 길이" if "지하부 길이" in growth_df.columns else None

            # --- 🥇 핵심 결과 카드 ---
            st.subheader("🥇 핵심 결과")
            if bio_col:
                bio_clean = growth_df[["개체번호", bio_col]].dropna()
                if len(bio_clean) >= 2:
                    avg_bio = bio_clean[bio_col].mean()
                    std_bio = bio_clean[bio_col].std()
                    cv_bio = (std_bio / avg_bio * 100) if avg_bio > 0 else np.nan
                    max_idx = bio_clean[bio_col].idxmax()
                    min_idx = bio_clean[bio_col].idxmin()
                    max_val = bio_clean.loc[max_idx, bio_col]
                    min_val = bio_clean.loc[min_idx, bio_col]
                    max_id = bio_clean.loc[max_idx, "개체번호"]
                    min_id = bio_clean.loc[min_idx, "개체번호"]

                    kc1, kc2, kc3, kc4 = st.columns(4)
                    kc1.metric("평균 생중량", f"{avg_bio:.2f} g")
                    kc2.metric("표준편차", f"{std_bio:.2f} g")
                    kc3.metric(
                        "CV (변동계수)",
                        f"{cv_bio:.1f}%" if not np.isnan(cv_bio) else "—",
                    )
                    kc4.metric(
                        "최대 ↔ 최소",
                        f"{max_val:.2f} g",
                        delta=f"최소 {min_val:.2f} g",
                    )
                    st.caption(
                        f"🏆 최대 생중량: 개체 {max_id} ({max_val:.2f} g)  |  "
                        f"최저 생중량: 개체 {min_id} ({min_val:.2f} g)"
                    )
                else:
                    st.info("📊 생중량 데이터가 충분하지 않아 핵심 지표를 계산할 수 없습니다.")
            else:
                st.info("📊 '생중량' 열이 없어 핵심 결과 카드를 표시할 수 없습니다.")

            # --- 개체별 생육 비교 (막대 그래프) ---
            st.subheader("📊 개체별 생육 비교")
            plot_cols = [
                c for c in ["생중량", "잎 수", "지상부 길이", "지하부 길이"]
                if c in growth_df.columns
            ]
            if plot_cols:
                top_threshold = None
                bot_threshold = None
                if bio_col:
                    bio_vals = growth_df[bio_col].dropna()
                    if len(bio_vals) >= 4:
                        top_threshold = bio_vals.quantile(0.75)
                        bot_threshold = bio_vals.quantile(0.25)

                for col in plot_cols:
                    col_data = (
                        growth_df[["개체번호", col]].dropna()
                        if "개체번호" in growth_df.columns
                        else growth_df[[col]].dropna()
                    )
                    if len(col_data) < 2:
                        continue
                    if col == bio_col and top_threshold is not None:
                        highlight = set()
                        for _, row in col_data.iterrows():
                            v = row[col]
                            if v >= top_threshold or v <= bot_threshold:
                                highlight.add(v)
                        fig_bar = bar_chart(
                            growth_df, col,
                            f"개체별 {col} 분포",
                            color="#4CAF50",
                            highlight_vals=highlight,
                        )
                    else:
                        fig_bar = bar_chart(
                            growth_df, col,
                            f"개체별 {col} 분포",
                            color="#4CAF50",
                        )
                    st.plotly_chart(fig_bar, width="stretch")
            else:
                st.info(
                    "📊 표시할 생육 측정 항목이 없습니다. "
                    "사이드바에서 항목을 선택하거나 CSV에 해당 열이 있는지 확인하세요."
                )

            # --- 생중량 분포 (박스플롯) ---
            # 고친 부분: go.Box에서 유효하지 않은 point_style 제거,
            # marker=dict() 형태로 개별 점 스타일 지정
            if bio_col:
                bio_for_box = growth_df[["개체번호", bio_col]].dropna()
                if len(bio_for_box) >= 3:
                    fig_box = go.Figure()
                    fig_box.add_trace(go.Box(
                        y=bio_for_box[bio_col],
                        name="생중량 분포",
                        fillcolor="#4CAF50",
                        boxpoints="all",
                        jitter=0.3,
                        pointpos=0,
                        marker=dict(
                            size=7,
                            color="#FF5722",
                            line=dict(width=1, color="white"),
                        ),
                        hovertemplate="%{y:.2f} g<extra></extra>",
                    ))
                    fig_box.update_layout(
                        title=dict(text="📦 생중량 분포 (박스플롯)", x=0.5),
                        yaxis_title="생중량 (g)",
                        showlegend=False,
                    )
                    apply_font(fig_box)
                    st.plotly_chart(fig_box, width="stretch")
                else:
                    st.info(
                        "📊 생중량 데이터가 3개체 미만이라 박스플롯을 표시할 수 없습니다."
                    )
            else:
                st.info("📊 '생중량' 열이 없어 분포 그래프를 표시할 수 없습니다.")

            # --- 상관관계 분석 (산점도 + 추세선) ---
            st.subheader("🔗 상관관계 분석 (산점도 + 추세선)")

            def scatter_with_trend(x_col, y_col, x_label, y_label):
                """np.polyfit 추세선을 직접 추가 (px.scatter trendline 사용 안 함)"""
                pair = growth_df[[x_col, y_col]].dropna()
                if len(pair) < 3:
                    return None
                x = pair[x_col].values
                y = pair[y_col].values
                r = np.corrcoef(x, y)[0, 1]
                slope, intercept = np.polyfit(x, y, 1)
                x_line = np.linspace(x.min(), x.max(), 100)
                y_line = slope * x_line + intercept
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=x,
                    y=y,
                    mode="markers",
                    marker=dict(
                        size=9,
                        color="#4CAF50",
                        line=dict(width=1, color="white"),
                    ),
                    hovertemplate=(
                        f"%{{x:.1f}} {x_label}<br>%{{y:.2f}} {y_label}<extra></extra>"
                    ),
                    name="데이터",
                ))
                fig.add_trace(go.Scatter(
                    x=x_line,
                    y=y_line,
                    mode="lines",
                    line=dict(color="#FF5722", width=2, dash="dash"),
                    name=f"추세선 (r = {r:.3f})",
                ))
                fig.update_layout(
                    title=dict(text=f"{x_label} vs {y_label}  (상관계수 r = {r:.3f})", x=0.5),
                    xaxis_title=x_label,
                    yaxis_title=y_label,
                    showlegend=True,
                )
                return apply_font(fig)

            scats = []
            if leaf_col and bio_col:
                scats.append((
                    "잎 수 vs 생중량", leaf_col, bio_col,
                    "잎 수 (장)", "생중량 (g)",
                ))
            if shoot_col and bio_col:
                scats.append((
                    "지상부 길이 vs 생중량", shoot_col, bio_col,
                    "지상부 길이 (mm)", "생중량 (g)",
                ))

            if scats:
                for title_s, x_c, y_c, xl, yl in scats:
                    fig_s = scatter_with_trend(x_c, y_c, xl, yl)
                    if fig_s:
                        st.plotly_chart(fig_s, width="stretch")
                    else:
                        st.info(
                            f"🔗 {title_s} 데이터가 충분하지 않아 "
                            "그래프를 표시할 수 없습니다."
                        )
            else:
                st.info(
                    "🔗 잎 수·지상부 길이·생중량 중 필요한 조합이 없어 "
                    "상관관계 분석을 표시할 수 없습니다."
                )

            # --- 상위 25% vs 하위 25% 비교 ---
            if bio_col:
                bio_vals = growth_df[bio_col].dropna()
                if len(bio_vals) >= 4:
                    t75 = bio_vals.quantile(0.75)
                    b25 = bio_vals.quantile(0.25)
                    top_df = growth_df[growth_df[bio_col] >= t75].dropna(subset=[bio_col])
                    bot_df = growth_df[growth_df[bio_col] <= b25].dropna(subset=[bio_col])

                    if len(top_df) >= 1 and len(bot_df) >= 1:
                        st.subheader("🏅 상위 25% vs 하위 25% 개체 비교")
                        comp_rows = []
                        comp_rows.append({
                            "그룹": "상위 25%",
                            "개체 수": len(top_df),
                            "평균 생중량 (g)": round(top_df[bio_col].mean(), 2),
                        })
                        comp_rows.append({
                            "그룹": "하위 25%",
                            "개체 수": len(bot_df),
                            "평균 생중량 (g)": round(bot_df[bio_col].mean(), 2),
                        })
                        for c in ["잎 수", "지상부 길이", "지하부 길이"]:
                            if c in growth_df.columns:
                                comp_rows.append({
                                    "그룹": "상위 25%",
                                    "개체 수": "",
                                    f"평균 {c}": round(top_df[c].mean(), 1),
                                })
                                comp_rows.append({
                                    "그룹": "하위 25%",
                                    "개체 수": "",
                                    f"평균 {c}": round(bot_df[c].mean(), 1),
                                })

                        st.dataframe(pd.DataFrame(comp_rows), width="stretch")
                        st.caption(
                            f"기준: 생중량 상위 25% ≥ {t75:.2f} g, "
                            f"하위 25% ≤ {b25:.2f} g"
                        )
                    else:
                        st.info(
                            "📊 상위/하위 25% 그룹이 각각 1개체 이상 필요합니다."
                        )
                else:
                    st.info(
                        "📊 생중량 데이터가 4개체 미만이라 "
                        "상위/하위 25% 비교를 할 수 없습니다."
                    )

            # --- 원본 + 다운로드 ---
            with st.expander("📄 생육 데이터 원본 테이블"):
                st.dataframe(growth_df, width="stretch")
                csv_gr = growth_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="📥 생육 데이터 CSV 다운로드",
                    data=csv_gr,
                    file_name=f"{school_name}_생육결과데이터.csv",
                    mime="text/csv",
                )
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")
