# main.py
import io
import os
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
#  한글 폰트 CSS
# ============================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
   font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
#  CSV 읽기 (utf-8-sig → cp949, bytes로 한 번만)
# ============================================================
@st.cache_data
def read_csv(file):
    data = file.getvalue()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()


# ============================================================
#  열 이름 표준화 (학교마다 다른 열 이름 → 표준 이름으로)
# ============================================================
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
    # 숫자 열 강제 변환 (글자+숫자 혼합 열 대응)
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
    # 생중량 없으면 지상부 질량 + 지하부 질량으로 계산
    if "생중량" not in df.columns and {"지상부 질량", "지하부 질량"} <= set(df.columns):
        df["생중량"] = df["지상부 질량"] + df["지하부 질량"]
    # 시간 열 파싱 (날짜_시각 형식도 지원)
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


# ============================================================
#  학교명 추출 (파일명 앞부분에서)
# ============================================================
def extract_school(filename):
    base = os.path.splitext(os.path.basename(filename))[0]
    cleaned = re.sub(r"[\d\-\._]+", "", base).strip()
    return cleaned if cleaned else "우리 학교"


# ============================================================
#  사이드바
# ============================================================
st.sidebar.header("📁 파일 업로드")
env_file = st.sidebar.file_uploader(
    "환경 데이터 CSV",
    type="csv",
    help="utf-8-sig 또는 cp949 인코딩 지원",
)
growth_file = st.sidebar.file_uploader(
    "생육 결과 데이터 CSV",
    type="csv",
    help="utf-8-sig 또는 cp949 인코딩 지원",
)
st.sidebar.markdown("---")
st.sidebar.subheader("📏 생육 측정 항목")

# ============================================================
#  데이터 로딩 (한 번만 읽고 재사용)
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
st.title("잎 갯수와 지하부 길이의 관련성")
if school_name and school_name != "우리 학교":
    st.caption(f"📍 {school_name} 극지식물 실험 데이터")


# ============================================================
#  3개 탭
# ============================================================
tab1, tab2, tab3 = st.tabs(["📋 실험 목적", "🌡️ 날짜별 pH", "📊 잎 수 vs 지하부 길이"])


# ------------------------------------------------------------
#  Tab 1: 실험 목적 + 지하부 길이 · pH
# ------------------------------------------------------------
with tab1:
    try:
        st.header("📋 실험 목적")
        st.markdown(
            f"""
**연구 배경**

극지식물(Extremophyte)은 건조, 고염, 낮은 온도 등 척박한 환경에서 살아남는 식물입니다.
이 실험은 **{school_name}**에서 재배한 극지식물 개체들을 대상으로,
개체마다 **잎 수**와 **지하부 길이**에 어떤 차이가 있는지,
그리고 이 두 생육 지표가 서로 어떤 관계에 있는지 알아보기 위해 진행되었습니다.

**통계 용어 짧은 설명**
- **평균(Mean)**: 모든 값을 더해 개수로 나눈 대표값
- **표준편차(Std)**: 값이 평균에서 얼마나 넓게 퍼져 있는지 나타내는 지표
- **변동계수(CV)**: 표준편차를 평균으로 나눈 값(%) — 크기가 다른 데이터끼리 비교할 때 유용
- **상관계수(r)**: 두 변수가 함께 움직이는 정도를 -1 ~ +1 사이 숫자로 표현한 값
  (+1에 가까울수록 함께 증가, -1에 가까울수록 함께 감소)
            """
        )

        if growth_df.empty and env_df.empty:
            st.info("📁 사이드바에서 CSV 파일을 업로드하면 분석 결과가 표시됩니다.")
        else:
            # --- 지하부 길이 정보 ---
            root_col = "지하부 길이" if "지하부 길이" in growth_df.columns else None
            st.subheader("🌱 개체의 지하부 길이")

            if root_col:
                root_clean = growth_df[[root_col]].dropna()
                if len(root_clean) >= 2:
                    mean_root = root_clean[root_col].mean()
                    std_root = root_clean[root_col].std()
                    cv_root = (std_root / mean_root * 100) if mean_root > 0 else np.nan
                    c1, c2, c3 = st.columns(3)
                    c1.metric(
                        "평균 지하부 길이",
                        f"{mean_root:.1f} mm",
                    )
                    c2.metric(
                        "표준편차",
                        f"{std_root:.1f} mm",
                    )
                    c3.metric(
                        "CV (변동계수)",
                        f"{cv_root:.1f}%"
                        if not np.isnan(cv_root)
                        else "—",
                    )
                    st.caption(
                        f"총 {len(growth_df)}개체 중 "
                        f"{len(root_clean)}개체의 지하부 길이 데이터가 있습니다."
                    )
                else:
                    st.info("📏 지하부 길이 데이터가 충분하지 않아 통계를 표시할 수 없습니다.")
            else:
                st.info("📏 CSV에 '지하부 길이' 열이 없어 지하부 길이 정보를 표시할 수 없습니다.")

            # --- pH 정보 ---
            st.subheader("🧪 환경 pH")
            ph_col = "ph" if "ph" in env_df.columns else None
            if ph_col:
                ph_clean = env_df[[ph_col]].dropna()
                if len(ph_clean) >= 2:
                    mean_ph = ph_clean[ph_col].mean()
                    std_ph = ph_clean[ph_col].std()
                    c4, c5 = st.columns(2)
                    c4.metric("평균 pH", f"{mean_ph:.2f}")
                    c5.metric("표준편차", f"{std_ph:.2f}")
                    st.caption(
                        f"총 {len(env_df)}개 측정값 중 "
                        f"{len(ph_clean)}개 pH 데이터가 있습니다."
                    )
                else:
                    st.info("🧪 pH 데이터가 충분하지 않아 통계를 표시할 수 없습니다.")
            else:
                st.info(
                    "🧪 CSV에 'ph' 열이 없어 pH 정보를 표시할 수 없습니다."
                )

            # --- 지하부 길이 분포 막대 그래프 ---
            if root_col:
                root_vals = growth_df[[root_col]].dropna()
                if len(root_vals) >= 2:
                    fig_root = px.bar(
                        root_vals.reset_index(drop=True),
                        y=root_col,
                        color=root_col,
                        color_continuous_scale=["#E8F5E9", "#2E7D32", "#1B5E20"],
                        labels={root_col: "지하부 길이 (mm)"},
                        title="개체별 지하부 길이 분포",
                    )
                    fig_root.update_layout(
                        yaxis_title="지하부 길이 (mm)",
                        showlegend=False,
                        font=dict(
                            family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"
                        ),
                        margin=dict(l=40, r=20, t=40, b=40),
                    )
                    fig_root.update_traces(
                        hovertemplate="%{y:.1f} mm<extra></extra>",
                    )
                    st.plotly_chart(fig_root, width="stretch")
                else:
                    st.info("📊 지하부 길이 데이터가 2개체 미만이라 그래프를 표시할 수 없습니다.")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")


# ------------------------------------------------------------
#  Tab 2: 날짜별 pH 막대 그래프
# ------------------------------------------------------------
with tab2:
    try:
        if env_df.empty:
            st.info(
                "📁 사이드바에서 환경 데이터 CSV를 업로드하면 날짜별 pH 그래프가 표시됩니다."
            )
        else:
            st.header("🌡️ 날짜별 pH값의 막대 그래프")
            time_col = "time" if "time" in env_df.columns else None
            ph_col = "ph" if "ph" in env_df.columns else None
            if time_col and ph_col:
                if not pd.api.types.is_datetime64_any_dtype(env_df[time_col]):
                    st.info(
                        "🌡️ 시간 열이 날짜 형식으로 인식되지 않아 "
                        "날짜별 집계를 할 수 없습니다."
                    )
                else:
                    env_ts = env_df[[time_col, ph_col]].dropna()
                    if len(env_ts) < 2:
                        st.info("🌡️ pH 데이터가 충분하지 않아 그래프를 표시할 수 없습니다.")
                    else:
                        env_ts = env_ts.copy()
                        env_ts["date"] = env_ts[time_col].dt.date
                        daily_ph = (
                            env_ts.groupby("date")[ph_col]
                            .mean()
                            .reset_index()
                            .rename(columns={ph_col: "pH"})
                            .sort_values("date")
                        )
                        if len(daily_ph) < 2:
                            st.info(
                                "🌡️ 날짜별로 집계된 pH 데이터가 충분하지 않습니다."
                            )
                        else:
                            fig_ph = px.bar(
                                daily_ph,
                                x="date",
                                y="pH",
                                color="pH",
                                color_continuous_scale=["#E0F7FA", "#00838F", "#004D40"],
                                labels={"date": "날짜", "pH": "pH"},
                                title="📅 날짜별 평균 pH값",
                            )
                            fig_ph.update_layout(
                                xaxis_title="날짜",
                                yaxis_title="pH",
                                showlegend=False,
                                font=dict(
                                    family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"
                                ),
                                margin=dict(l=40, r=20, t=40, b=40),
                            )
                            fig_ph.update_traces(
                                hovertemplate="%{x}<br>pH: %{y:.2f}<extra></extra>",
                            )
                            st.plotly_chart(fig_ph, width="stretch")
            else:
                if not ph_col:
                    st.info("🌡️ CSV에 'ph' 열이 없어 pH 그래프를 표시할 수 없습니다.")
                if not time_col:
                    st.info(
                        "🌡️ CSV에 'time' 열이 없거나 날짜 형식이 아니라 "
                        "날짜별 그래프를 표시할 수 없습니다."
                    )

        # 환경 데이터 원본 + 다운로드
        with st.expander("📄 환경 데이터 원본 테이블"):
            if not env_df.empty:
                st.dataframe(env_df, width="stretch")
                csv_env = env_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="📥 환경 데이터 CSV 다운로드",
                    data=csv_env,
                    file_name=f"{school_name}_환경데이터.csv",
                    mime="text/csv",
                )
            else:
                st.info("📄 환경 데이터가 없습니다.")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")


# ------------------------------------------------------------
#  Tab 3: 잎 수 vs 지하부 길이 산점도 + 추세선
# ------------------------------------------------------------
with tab3:
    try:
        if growth_df.empty:
            st.info(
                "📁 사이드바에서 생육 결과 데이터 CSV를 업로드하면 산점도가 표시됩니다."
            )
        else:
            st.header("📊 잎 수와 지하부 길이의 산점도")
            leaf_col = "잎 수" if "잎 수" in growth_df.columns else None
            root_col = "지하부 길이" if "지하부 길이" in growth_df.columns else None

            if leaf_col and root_col:
                pair = growth_df[[leaf_col, root_col]].dropna()
                if len(pair) < 3:
                    st.info(
                        "📊 잎 수와 지하부 길이 데이터가 3개체 미만이라 "
                        "산점도를 표시할 수 없습니다."
                    )
                else:
                    x = pair[leaf_col].values
                    y = pair[root_col].values
                    r = np.corrcoef(x, y)[0, 1]
                    slope, intercept = np.polyfit(x, y, 1)
                    x_line = np.linspace(x.min(), x.max(), 100)
                    y_line = slope * x_line + intercept

                    fig_scatter = px.scatter(
                        pair,
                        x=leaf_col,
                        y=root_col,
                        labels={leaf_col: "잎 수 (장)", root_col: "지하부 길이 (mm)"},
                        title=(
                            f"잎 수 vs 지하부 길이  "
                            f"(상관계수 r = {r:.3f})"
                        ),
                    )
                    # 추세선 직접 추가 (px.scatter trendline 사용 안 함)
                    fig_scatter.add_scatter(
                        x=x_line,
                        y=y_line,
                        mode="lines",
                        line=dict(color="#E64A19", width=2, dash="dash"),
                        name=f"추세선 (r = {r:.3f})",
                        hoverinfo="skip",
                    )
                    fig_scatter.update_traces(
                        marker=dict(
                            size=10,
                            color="#2E7D32",
                            line=dict(width=1, color="white"),
                        ),
                        selector=dict(mode="markers"),
                        hovertemplate=(
                            f"%{{x:.0f}}장<br>%{{y:.1f}}mm<extra></extra>"
                        ),
                    )
                    fig_scatter.update_layout(
                        xaxis_title="잎 수 (장)",
                        yaxis_title="지하부 길이 (mm)",
                        title=dict(
                            text=(
                                f"잎 수 vs 지하부 길이  "
                                f"(상관계수 r = {r:.3f})"
                            ),
                            x=0.5,
                        ),
                        showlegend=True,
                        font=dict(
                            family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"
                        ),
                        margin=dict(l=40, r=20, t=50, b=40),
                    )
                    st.plotly_chart(fig_scatter, width="stretch")
                    st.caption(
                        f"상관계수 r = {r:.3f}  |  "
                        f"추세선: 지하부 길이 = {slope:.2f} × 잎 수 + {intercept:.2f}"
                    )
            else:
                missing = []
                if not leaf_col:
                    missing.append("잎 수")
                if not root_col:
                    missing.append("지하부 길이")
                st.info(
                    f"📊 CSV에 {'·'.join(missing)} 열이 없어 산점도를 표시할 수 없습니다."
                )

            # --- 상위 25% vs 하위 25% 비교 ---
            if leaf_col and root_col:
                leaf_vals = growth_df[leaf_col].dropna()
                root_vals = growth_df[root_col].dropna()
                if len(leaf_vals) >= 4 and len(root_vals) >= 4:
                    leaf_t75 = leaf_vals.quantile(0.75)
                    leaf_b25 = leaf_vals.quantile(0.25)
                    root_t75 = root_vals.quantile(0.75)
                    root_b25 = root_vals.quantile(0.25)

                    top_leaves = growth_df[growth_df[leaf_col] >= leaf_t75].dropna(
                        subset=[leaf_col]
                    )
                    bot_leaves = growth_df[growth_df[leaf_col] <= leaf_b25].dropna(
                        subset=[leaf_col]
                    )
                    top_roots = growth_df[growth_df[root_col] >= root_t75].dropna(
                        subset=[root_col]
                    )
                    bot_roots = growth_df[growth_df[root_col] <= root_b25].dropna(
                        subset=[root_col]
                    )

                    if len(top_leaves) >= 1 and len(bot_leaves) >= 1:
                        st.subheader("🏅 잎 수 기준 상위 25% vs 하위 25%")
                        rows_leaf = []
                        rows_leaf.append({
                            "그룹": "잎 수 상위 25%",
                            "개체 수": len(top_leaves),
                            "평균 잎 수 (장)": round(top_leaves[leaf_col].mean(), 1),
                            "평균 지하부 길이 (mm)": round(
                                top_leaves[root_col].mean(), 1
                            )
                            if root_col in top_leaves.columns
                            else "—",
                        })
                        rows_leaf.append({
                            "그룹": "잎 수 하위 25%",
                            "개체 수": len(bot_leaves),
                            "평균 잎 수 (장)": round(bot_leaves[leaf_col].mean(), 1),
                            "평균 지하부 길이 (mm)": round(
                                bot_leaves[root_col].mean(), 1
                            )
                            if root_col in bot_leaves.columns
                            else "—",
                        })
                        st.dataframe(pd.DataFrame(rows_leaf), width="stretch")
                        st.caption(
                            f"잎 수 기준: 상위 25% ≥ {leaf_t75:.0f}장, "
                            f"하위 25% ≤ {leaf_b25:.0f}장"
                        )

                    if len(top_roots) >= 1 and len(bot_roots) >= 1:
                        st.subheader("🏅 지하부 길이 기준 상위 25% vs 하위 25%")
                        rows_root = []
                        rows_root.append({
                            "그룹": "지하부 길이 상위 25%",
                            "개체 수": len(top_roots),
                            "평균 잎 수 (장)": round(
                                top_roots[leaf_col].mean(), 1
                            )
                            if leaf_col in top_roots.columns
                            else "—",
                            "평균 지하부 길이 (mm)": round(
                                top_roots[root_col].mean(), 1
                            ),
                        })
                        rows_root.append({
                            "그룹": "지하부 길이 하위 25%",
                            "개체 수": len(bot_roots),
                            "평균 잎 수 (장)": round(
                                bot_roots[leaf_col].mean(), 1
                            )
                            if leaf_col in bot_roots.columns
                            else "—",
                            "평균 지하부 길이 (mm)": round(
                                bot_roots[root_col].mean(), 1
                            ),
                        })
                        st.dataframe(pd.DataFrame(rows_root), width="stretch")
                        st.caption(
                            f"지하부 길이 기준: 상위 25% ≥ {root_t75:.0f}mm, "
                            f"하위 25% ≤ {root_b25:.0f}mm"
                        )

            # --- 생육 데이터 원본 + 다운로드 ---
            with st.expander("📄 생육 데이터 원본 테이블"):
                if not growth_df.empty:
                    st.dataframe(growth_df, width="stretch")
                    csv_gr = growth_df.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        label="📥 생육 데이터 CSV 다운로드",
                        data=csv_gr,
                        file_name=f"{school_name}_생육결과데이터.csv",
                        mime="text/csv",
                    )
                else:
                    st.info("📄 생육 데이터가 없습니다.")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")
