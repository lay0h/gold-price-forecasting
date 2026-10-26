import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_loader import download_data, get_clean
from eda import add_features, correlation_table, outlier_summary
from Models import run_models

st.set_page_config(page_title="Gold Price Forecasting", page_icon="🪙", layout="wide")

# ------------------------------------------------------------------ palette
GOLD, LIGHT_GOLD, DEEP_GOLD = "#D4AF37", "#F4D878", "#8A6D1D"
CHAMP, WARM, MUTED, LOSS, EMERALD = "#E8D7A7", "#F5F2EA", "#BDB5A5", "#C4574A", "#2FA37F"
MODEL_COLORS = {"Naive": "#A8A8A8", "Linear Regression": CHAMP, "Random Forest": GOLD}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Playfair+Display:wght@500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif}
.stApp{background:radial-gradient(circle at 8% 0%,rgba(255,252,240,.75),transparent 40%),radial-gradient(circle at 95% 30%,rgba(212,175,55,.28),transparent 38%),repeating-linear-gradient(115deg,rgba(255,255,255,.05) 0 2px,transparent 2px 10px),linear-gradient(135deg,#F3E8C6 0%,#E8D7A7 50%,#D9C48C 100%);color:#181818}
header[data-testid="stHeader"]{background:transparent}
.block-container{max-width:1320px;padding-top:1.6rem;padding-bottom:4rem}
@keyframes rise{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:none}}
@keyframes sheen{to{background-position:300% center}}
@keyframes sway{0%,100%{transform:rotateX(-24deg) rotateY(-34deg)}50%{transform:rotateX(-18deg) rotateY(-10deg)}}
@keyframes coin{to{transform:rotateY(360deg)}}
@keyframes twinkle{0%,100%{opacity:.15;transform:scale(.6)}50%{opacity:.95;transform:scale(1.2)}}
@keyframes glint{0%{transform:translateX(-130%) skewX(-20deg)}60%,100%{transform:translateX(330%) skewX(-20deg)}}
@keyframes pulse{0%,100%{box-shadow:0 0 0 0 rgba(212,175,55,.7)}50%{box-shadow:0 0 0 9px rgba(212,175,55,0)}}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#1b1915,#14130f);border-right:1px solid rgba(212,175,55,.45)}
section[data-testid="stSidebar"] *{color:#F5F2EA!important}
section[data-testid="stSidebar"] h2,section[data-testid="stSidebar"] h3{font-family:'Playfair Display',serif;color:#F4D878!important}
section[data-testid="stSidebar"] [data-baseweb="select"]>div{background:#2a2721!important;border:1px solid rgba(212,175,55,.45)!important}
section[data-testid="stSidebar"] span[data-baseweb="tag"]{background:#D4AF37!important}
section[data-testid="stSidebar"] span[data-baseweb="tag"] *{color:#181818!important}
section[data-testid="stSidebar"] div[role="slider"]{background:#D4AF37!important}
div[data-baseweb="popover"] ul,div[data-baseweb="popover"] li{background:#2a2721!important;color:#F5F2EA!important}
div[data-baseweb="popover"] li:hover{background:#4a3f1c!important}
div[data-baseweb="tab-list"]{gap:.5rem;border-bottom:2px solid rgba(24,24,24,.25)}
button[data-baseweb="tab"]{background:rgba(24,24,24,.07);border-radius:14px 14px 0 0;padding:.7rem 1.5rem}
button[data-baseweb="tab"] p{color:#181818!important;font-weight:600;font-size:1.05rem}
button[data-baseweb="tab"][aria-selected="true"]{background:#1c1a17}
button[data-baseweb="tab"][aria-selected="true"] p{color:#F4D878!important}
div[data-baseweb="tab-highlight"]{background:#D4AF37!important;height:3px!important}
.hero{position:relative;overflow:hidden;min-height:290px;border-radius:30px;padding:2.8rem 3rem;margin-bottom:1.5rem;border:1px solid rgba(244,216,120,.6);background:linear-gradient(115deg,#1c1a17 0%,#26221a 55%,#5b4510 100%);box-shadow:0 24px 60px rgba(60,42,6,.45);display:flex;align-items:center;justify-content:space-between;gap:1rem;animation:rise .8s ease both}
.hero:after{content:"";position:absolute;inset:0;pointer-events:none;background:radial-gradient(circle at 78% 35%,rgba(244,216,120,.28),transparent 32%),linear-gradient(120deg,transparent 20%,rgba(255,255,255,.05) 35%,transparent 48%)}
.hero-copy{position:relative;z-index:2;max-width:700px}
.hero-top{display:flex;align-items:center;gap:1rem;margin-bottom:.9rem}
.eyebrow{color:#F4D878;text-transform:uppercase;font-size:.8rem;letter-spacing:.26em;font-weight:700}
.hero h1{font-family:'Playfair Display',serif;font-size:4rem;line-height:1.04;margin:0;padding:0;background:linear-gradient(90deg,#FFF9E8 0%,#F4D878 50%,#D4AF37 100%);background-size:200% auto;-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent;animation:sheen 12s linear infinite}
.hero p{color:#EFE6D0;font-size:1.1rem;line-height:1.7;margin-top:1rem;max-width:640px}
.scene{perspective:1000px;width:320px;height:230px;flex-shrink:0;position:relative;z-index:2}
.stack{position:absolute;left:50%;top:55%;width:0;height:0;transform-style:preserve-3d;animation:sway 9s ease-in-out infinite}
.ingot{position:absolute;width:120px;height:34px;left:-60px;top:-17px;transform-style:preserve-3d}
.fa{position:absolute;display:block;overflow:hidden}
.fr{width:120px;height:34px;transform:translateZ(28px);background:linear-gradient(135deg,#F8EBB5 0%,#E3C766 30%,#D4AF37 55%,#9A7418 100%)}
.fr:after{content:"";position:absolute;top:0;bottom:0;width:30px;background:rgba(255,255,255,.55);animation:glint 5s ease-in-out infinite}
.fr b{display:block;text-align:center;font:700 11px/34px 'Playfair Display',serif;letter-spacing:.22em;color:#5a410a}
.bk{width:120px;height:34px;transform:rotateY(180deg) translateZ(28px);background:#7a5c14}
.rt{width:56px;height:34px;left:32px;transform:rotateY(90deg) translateZ(60px);background:linear-gradient(180deg,#B8942A,#6B510F)}
.lf{width:56px;height:34px;left:32px;transform:rotateY(-90deg) translateZ(60px);background:linear-gradient(180deg,#B8942A,#6B510F)}
.tp{width:120px;height:56px;top:-11px;transform:rotateX(90deg) translateZ(17px);background:linear-gradient(135deg,#FFF3BF,#E8CB6A)}
.bt{width:120px;height:56px;top:-11px;transform:rotateX(-90deg) translateZ(17px);background:#5c4710}
.coin3d{position:relative;width:58px;height:58px;transform-style:preserve-3d;animation:coin 5s linear infinite;flex-shrink:0}
.coin3d i{position:absolute;inset:0;border-radius:50%;backface-visibility:hidden;display:flex;align-items:center;justify-content:center;font:700 1.15rem 'Playfair Display',serif;color:#5a410a;border:3px solid #F3E3A0;background:radial-gradient(circle at 30% 30%,#FFF3BF,#D4AF37 55%,#8A6D1D);box-shadow:0 0 18px rgba(244,216,120,.55);font-style:normal}
.coin3d i+i{transform:rotateY(180deg)}
.sp{position:absolute;width:7px;height:7px;border-radius:50%;background:#F4D878;animation:twinkle 3.2s ease-in-out infinite;z-index:1}
.sec{margin:2.4rem 0 1rem}
.sec-t{font-family:'Playfair Display',serif;font-size:2.3rem;font-weight:700;color:#181818;line-height:1.15}
.sec-s{color:#3b3216;font-size:1.05rem;margin-top:.25rem}
.gline{height:3px;margin-top:.7rem;border-radius:3px;background:linear-gradient(90deg,#181818,#D4AF37 45%,transparent)}
.rel-l{color:#6d5209;text-transform:uppercase;font-size:.8rem;font-weight:800;letter-spacing:.16em;margin-top:2rem}
.rel-t{font-family:'Playfair Display',serif;color:#181818;font-size:1.65rem;font-weight:700;margin:.1rem 0 .2rem}
.rel-d{color:#4a3f1c;font-size:.98rem;margin-bottom:.9rem}
.gk{position:relative;overflow:hidden;min-height:176px;border-radius:24px;padding:1.3rem 1.4rem;box-shadow:inset 0 1px 0 rgba(255,255,255,.5),0 16px 34px rgba(60,42,6,.35);transition:.35s;animation:rise .8s ease both;transform-style:preserve-3d}
.gk:hover{transform:perspective(800px) rotateX(5deg) translateY(-6px);box-shadow:0 24px 44px rgba(60,42,6,.45)}
.gk>div{position:relative;z-index:1}
.gk-l{font-size:.78rem;text-transform:uppercase;letter-spacing:.14em;font-weight:800;opacity:.88}
.gk-v{font-family:'Playfair Display',serif;font-size:2.5rem;font-weight:800;margin-top:.3rem;line-height:1.1}
.gk-s{font-size:.9rem;font-weight:600;margin-top:.3rem;opacity:.92}
.gk .spark{position:absolute;left:0;bottom:0;width:100%;height:64px;z-index:0}
.gk-gold{background:linear-gradient(135deg,#F9EAA9 0%,#D4AF37 55%,#A67C1A 100%);color:#1d1606}
.gk-rose{background:linear-gradient(135deg,#F6E3C4 0%,#E0A98A 55%,#B9706B 100%);color:#2b1408}
.gk-emerald{background:linear-gradient(135deg,#3DB88F 0%,#14705A 60%,#0B4A3A 100%);color:#FFFFFF}
.gk-graph{background:linear-gradient(135deg,#4a4132 0%,#262219 60%,#14130f 100%);color:#F5F2EA}
.stat{background:linear-gradient(160deg,#25221b,#181612);border:1px solid rgba(212,175,55,.4);border-radius:20px;padding:1.2rem 1.35rem;box-shadow:0 12px 28px rgba(60,42,6,.28);height:100%;animation:rise .8s ease both}
.stat:before{content:"";display:block;height:3px;width:46px;border-radius:3px;background:linear-gradient(90deg,#8A6D1D,#F4D878);margin-bottom:.8rem}
.stat-l{color:#E8D7A7;font-size:.78rem;text-transform:uppercase;letter-spacing:.14em;font-weight:700}
.stat-v{font-family:'Playfair Display',serif;font-size:2.6rem;font-weight:800;color:#FFF9E8;line-height:1.1;margin-top:.3rem}
.stat-v small{font-family:'Inter',sans-serif;font-size:1rem;color:#F4D878;font-weight:600;margin-left:.3rem}
.stat-v.sm{font-size:1.65rem;line-height:1.35}
.stat-c{color:#CFC7B4;font-size:.9rem;margin-top:.35rem}
.ingot-card{position:relative;overflow:hidden;min-height:148px;border-radius:18px;padding:1.2rem 1.3rem;color:#1d1606;background:linear-gradient(145deg,#FBEFB8 0%,#E3C766 28%,#C79A27 60%,#8D6717 100%);box-shadow:inset 0 3px 0 rgba(255,255,255,.65),inset 0 -6px 0 rgba(90,64,10,.5),0 16px 30px rgba(60,42,6,.4);transition:.3s;animation:rise .8s ease both}
.ingot-card:hover{transform:perspective(700px) rotateX(7deg) translateY(-5px)}
.ingot-card:after{content:"";position:absolute;top:0;bottom:0;width:50px;background:rgba(255,255,255,.4);animation:glint 6s ease-in-out infinite}
.ig-l{font-size:.78rem;text-transform:uppercase;letter-spacing:.14em;font-weight:800}
.ig-v{font-family:'Playfair Display',serif;font-size:2.2rem;font-weight:800;margin-top:.35rem;line-height:1.1}
.ig-s{font-size:.9rem;font-weight:700;margin-top:.35rem;opacity:.85}
div[class*="st-key-card_"]{background:linear-gradient(160deg,#25221b,#181612);border:1px solid rgba(212,175,55,.38);border-radius:20px;padding:1.3rem 1.5rem;box-shadow:0 12px 32px rgba(60,42,6,.3);margin-bottom:1.1rem;animation:rise .7s ease both}
div[class*="st-key-card_"] p,div[class*="st-key-card_"] span,div[class*="st-key-card_"] label,div[class*="st-key-card_"] li,div[class*="st-key-card_"] h4{color:#F5F2EA!important}
div[class*="st-key-card_"] button[data-testid*="pills"]{background:#322e24!important;border:1px solid rgba(212,175,55,.55)!important}
div[class*="st-key-card_"] button[data-testid*="pills"] *{color:#F5F2EA!important}
div[class*="st-key-card_"] button[data-testid="stBaseButton-pillsActive"],div[class*="st-key-card_"] button[aria-checked="true"]{background:#D4AF37!important;border-color:#F4D878!important}
div[class*="st-key-card_"] button[data-testid="stBaseButton-pillsActive"] *,div[class*="st-key-card_"] button[aria-checked="true"] *{color:#181818!important;font-weight:700!important}
.ct{font-family:'Playfair Display',serif;font-size:1.4rem;color:#F4D878;font-weight:700}
.cs{color:#CFC7B4;font-size:.92rem;margin:.1rem 0 .5rem}
.note{margin-top:.9rem;padding:.95rem 1.15rem;border-left:4px solid #D4AF37;background:#2f2b21;border-radius:0 14px 14px 0;color:#F5F2EA;font-size:1.02rem;line-height:1.65}
.note .nb{display:inline-block;background:#D4AF37;color:#181818;font-size:.68rem;font-weight:800;letter-spacing:.14em;padding:.18rem .6rem;border-radius:20px;margin-right:.7rem;vertical-align:middle}
.note b{color:#F4D878}
.chip{display:inline-block;margin:.25rem .35rem .25rem 0;padding:.4rem .95rem;border-radius:20px;background:linear-gradient(135deg,#F4D878,#C79A27);color:#1d1606;font-weight:700;font-size:.9rem;box-shadow:0 4px 10px rgba(0,0,0,.3)}
.step{background:rgba(255,255,255,.04);border:1px solid rgba(212,175,55,.25);border-radius:16px;padding:1rem 1.1rem;height:100%}
.step-n{color:#F4D878;font-size:.78rem;letter-spacing:.14em;text-transform:uppercase;font-weight:800}
.step-t{font-family:'Playfair Display',serif;color:#FFF9E8;font-size:1.25rem;font-weight:700;margin:.3rem 0}
.step-d{color:#D9D1C2;font-size:.92rem;line-height:1.55}
.mc-n{font-size:1.9rem}
"""
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


# ------------------------------------------------------------------ helpers
def esc(t):  # '$' would otherwise trigger LaTeX in st.markdown
    return str(t).replace("$", "&#36;")


def money(v):
    return f"${v:,.2f}"


def sec(title, sub=""):
    st.markdown(f'<div class="sec"><div class="sec-t">{esc(title)}</div><div class="sec-s">{esc(sub)}</div><div class="gline"></div></div>', unsafe_allow_html=True)


def rel(label, title, desc=""):
    st.markdown(f'<div class="rel-l">{esc(label)}</div><div class="rel-t">{esc(title)}</div><div class="rel-d">{esc(desc)}</div>', unsafe_allow_html=True)


_cards = [0]


def panel(title=None, sub=None):
    _cards[0] += 1
    try:
        c = st.container(key=f"card_{_cards[0]}")
    except TypeError:
        c = st.container()
    if title:
        c.markdown(f'<div class="ct">{esc(title)}</div>' + (f'<div class="cs">{esc(sub)}</div>' if sub else ""), unsafe_allow_html=True)
    return c


def note(text):
    st.markdown(f'<div class="note"><span class="nb">INSIGHT</span>{esc(text)}</div>', unsafe_allow_html=True)


def stat(label, value, unit="", cap="", small=False):
    u = f"<small>{esc(unit)}</small>" if unit else ""
    return f'<div class="stat"><div class="stat-l">{esc(label)}</div><div class="stat-v{" sm" if small else ""}">{esc(value)}{u}</div><div class="stat-c">{esc(cap)}</div></div>'


def ingot_card(label, value, sub=""):
    return f'<div class="ingot-card"><div class="ig-l">{esc(label)}</div><div class="ig-v">{esc(value)}</div><div class="ig-s">{esc(sub)}</div></div>'


def spark(vals, line="#FFFFFF", shadow="rgba(0,0,0,.28)", w=300, h=64):
    v = np.asarray(pd.Series(vals).dropna().tail(70), dtype=float)
    if len(v) < 2:
        return ""
    lo, hi = v.min(), v.max()
    xs = np.linspace(0, w, len(v)); ys = h - 8 - (v - lo) / ((hi - lo) or 1) * (h - 24)
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    return (f'<svg class="spark" viewBox="0 0 {w} {h}" preserveAspectRatio="none"><polygon points="0,{h} {pts} {w},{h}" fill="{line}" opacity=".16"/>'
            f'<polyline points="{pts}" fill="none" stroke="{shadow}" stroke-width="7" stroke-linejoin="round" stroke-linecap="round" transform="translate(0,4)"/>'
            f'<polyline points="{pts}" fill="none" stroke="{line}" stroke-width="3.5" stroke-linejoin="round" stroke-linecap="round"/></svg>')


def gk(cls, label, value, sub, series, line="#FFFFFF"):
    return f'<div class="gk {cls}"><div class="gk-l">{esc(label)}</div><div class="gk-v">{esc(value)}</div><div class="gk-s">{esc(sub)}</div>{spark(series, line)}</div>'


def ingot3d(x, y, label):
    return (f'<div class="ingot" style="transform:translate3d({x}px,{y}px,0)"><i class="fa fr"><b>{label}</b></i><i class="fa bk"></i><i class="fa rt"></i>'
            '<i class="fa lf"></i><i class="fa tp"></i><i class="fa bt"></i></div>')


def style(fig, h=380):
    fig.update_layout(template=None, height=h, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.03)",
                      font=dict(color=WARM, family="Inter", size=13), margin=dict(l=20, r=20, t=30, b=25),
                      colorway=[GOLD, CHAMP, "#A8A8A8", EMERALD, LOSS], legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=WARM, size=13)),
                      hoverlabel=dict(bgcolor="#322e24", font_color=WARM, font_size=13))
    fig.update_xaxes(gridcolor="rgba(255,255,255,.09)", zerolinecolor="rgba(255,255,255,.18)", linecolor="rgba(255,255,255,.2)",
                     tickfont=dict(color=WARM, size=12), title_font=dict(color=CHAMP, size=13))
    fig.update_yaxes(gridcolor="rgba(255,255,255,.09)", zerolinecolor="rgba(255,255,255,.18)", linecolor="rgba(255,255,255,.2)",
                     tickfont=dict(color=WARM, size=12), title_font=dict(color=CHAMP, size=13))
    try:
        fig.update_layout(barcornerradius=8)
    except Exception:
        pass
    return fig


def plot(fig, h=380):
    style(fig, h)
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def table(df, **kw):
    try:
        st.dataframe(df, width="stretch", **kw)
    except TypeError:
        st.dataframe(df, use_container_width=True, **kw)


def area_trace(x, y, name, color=GOLD):
    try:
        return go.Scatter(x=x, y=y, name=name, line=dict(color=LIGHT_GOLD, width=2.4), fill="tozeroy",
                          fillgradient=dict(type="vertical", colorscale=[[0, "rgba(212,175,55,0)"], [1, "rgba(212,175,55,.55)"]]))
    except Exception:
        return go.Scatter(x=x, y=y, name=name, line=dict(color=LIGHT_GOLD, width=2.4), fill="tozeroy", fillcolor="rgba(212,175,55,.2)")


# ------------------------------------------------------------------ data
@st.cache_data(show_spinner="Loading gold market data...")
def load_all_data():
    raw = download_data()
    raw.columns = raw.columns.str.strip().str.lower().str.replace(" ", "_")
    if "date" in raw.columns:
        raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    clean, report = get_clean()
    return raw, clean, add_features(clean), report


@st.cache_resource(show_spinner="Training prediction models...")
def train_models(data):
    return run_models(data)


raw_df, clean_df, full_df, report = load_all_data()
ml = train_models(full_df)

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown('<div style="display:flex;align-items:center;gap:.8rem"><div class="coin3d"><i>Au</i><i>999</i></div><div style="font-family:Playfair Display,serif;font-size:1.5rem;color:#F4D878;font-weight:700">Gold Analytics</div></div>', unsafe_allow_html=True)
    st.caption("About the data: daily XAU/USD prices (Open, High, Low, Close, Volume) from Kaggle. The app explores gold's behaviour and forecasts the next closing price.")
    st.markdown("### Filters")
    y_min, y_max = int(full_df["year"].min()), int(full_df["year"].max())
    year_range = st.slider("Year range", y_min, y_max, (y_min, y_max))
    wd_all = [w for w in WEEKDAYS if w in set(full_df["weekday"].dropna())]
    weekdays = st.multiselect("Weekdays", wd_all, default=wd_all)
    ma_choice = st.selectbox("Moving average line", ["20-day (ma_20)", "5-day (ma_5)"])
    ma_col = "ma_20" if ma_choice.startswith("20") else "ma_5"
    st.caption("Filters update the Overview cards and EDA. Machine-learning evaluation always uses the full prepared dataset.")

filtered = full_df[full_df["year"].between(*year_range) & full_df["weekday"].isin(weekdays)].copy()
if len(filtered) < 30:
    st.warning("The selected filters leave too little data. Please widen them.")
    st.stop()

best_mae_row = ml["metrics"].loc[ml["metrics"]["MAE"].idxmin()]
ml_only = ml["metrics"][ml["metrics"]["Model"] != "Naive"]
best_ml_row = ml_only.loc[ml_only["MAE"].idxmin()]
fc_tbl = ml["forecast"]
fc_val = float(fc_tbl.loc[fc_tbl["Model"] == best_ml_row["Model"], "Forecast"].iloc[0])
last_close_full = float(full_df["close"].iloc[-1])

latest = filtered.iloc[-1]; highest = filtered.loc[filtered["high"].idxmax()]; lowest = filtered.loc[filtered["low"].idxmin()]

tab1, tab2, tab3 = st.tabs(["Overview", "EDA", "Machine Learning"])

# ================================================================== OVERVIEW
with tab1:
    stack = ingot3d(-66, 44, "999.9") + ingot3d(66, 44, "FINE") + ingot3d(0, 8, "XAU")
    sparkles = "".join(f'<span class="sp" style="left:{x}%;top:{y}%;animation-delay:{d}s"></span>' for x, y, d in [(52, 18, 0), (63, 70, .8), (71, 24, 1.6), (86, 60, 2.2), (92, 20, .4), (45, 78, 1.2)])
    st.markdown(f'<div class="hero">{sparkles}<div class="hero-copy"><div class="hero-top"><div class="coin3d"><i>Au</i><i>999</i></div><div class="eyebrow">Market Analytics · Machine Learning · XAU/USD</div></div>'
                '<h1>Gold Price Forecasting</h1><p>Exploring trends, volatility, and machine learning predictions for XAU/USD, from the raw market file to cleaned features, outlier review and next-day forecasts.</p></div>'
                f'<div class="scene"><div class="stack">{stack}</div></div></div>', unsafe_allow_html=True)

    k = st.columns(4)
    k[0].markdown(gk("gk-gold", "Latest Close", money(latest["close"]), latest["date"].strftime("%d %b %Y"), filtered["close"]), unsafe_allow_html=True)
    k[1].markdown(gk("gk-rose", "Highest Price", money(highest["high"]), highest["date"].strftime("%d %b %Y"), filtered["high"]), unsafe_allow_html=True)
    k[2].markdown(gk("gk-emerald", "Lowest Price", money(lowest["low"]), lowest["date"].strftime("%d %b %Y"), filtered["low"], "#F9EAA9"), unsafe_allow_html=True)
    chg = (fc_val / last_close_full - 1) * 100
    k[3].markdown(gk("gk-graph", "Predicted Next Close", money(fc_val), f"{chg:+.2f}% · {best_ml_row['Model']}", pd.concat([filtered["close"], pd.Series([fc_val])]), "#F4D878"), unsafe_allow_html=True)

    # ---------------- A. RAW
    sec(" Raw Data Overview", "The source file exactly as downloaded from Kaggle, before any cleaning.")
    has_date = "date" in raw_df.columns and raw_df["date"].notna().any()
    cover = f"{raw_df['date'].min():%b %Y} – {raw_df['date'].max():%b %Y}" if has_date else "n/a"
    a = st.columns(4)
    a[0].markdown(stat("Raw rows", f"{raw_df.shape[0]:,}", cap="One row per trading day"), unsafe_allow_html=True)
    a[1].markdown(stat("Raw columns", f"{raw_df.shape[1]}", cap="As found in the file"), unsafe_allow_html=True)
    a[2].markdown(stat("Coverage", cover, cap="First to last date", small=True), unsafe_allow_html=True)
    a[3].markdown(stat("Frequency", "Daily", cap="XAU/USD, one candle per day", small=True), unsafe_allow_html=True)
    st.write("")
    with panel("Raw columns", "Column names in the original file"):
        st.markdown("".join(f'<span class="chip">{c}</span>' for c in raw_df.columns), unsafe_allow_html=True)
    with panel("Preview of raw data", "First 10 rows of the original file"):
        table(raw_df.head(10))
    c1, c2 = st.columns(2)
    with c1:
        with panel("Data types", "Type and number of non-null values per column"):
            table(pd.DataFrame({"Column": raw_df.columns, "Data Type": raw_df.dtypes.astype(str).values, "Non-Null": raw_df.notnull().sum().values}), hide_index=True, height=290)
    with c2:
        with panel("Summary statistics", "Numeric columns of the raw file"):
            rn = raw_df.select_dtypes(include=np.number)
            if not rn.empty:
                table(rn.describe().T.round(3), height=290)
    with panel():
        note(f"The raw file holds {raw_df.shape[0]:,} daily rows and {raw_df.shape[1]} columns. This is the baseline used to show exactly what changed during cleaning and feature engineering.")

    # ---------------- B. CLEANING
    original_cols = list(raw_df.columns)
    added_cols = [c for c in full_df.columns if c not in original_cols]
    removed = raw_df.shape[0] - clean_df.shape[0]
    sec(" Cleaning & Feature Engineering Summary", "What was removed, what was added, and why.")
    b = st.columns(3)
    b[0].markdown(stat("Rows before cleaning", f"{raw_df.shape[0]:,}", cap="Raw file"), unsafe_allow_html=True)
    b[1].markdown(stat("Rows after cleaning", f"{clean_df.shape[0]:,}", cap="Clean project data"), unsafe_allow_html=True)
    b[2].markdown(stat("Rows removed", f"{removed:,}", unit=f"({removed / raw_df.shape[0] * 100:.1f}%)", cap="Mainly older history outside the project scope"), unsafe_allow_html=True)
    st.write("")
    b = st.columns(3)
    b[0].markdown(stat("Original columns", f"{len(original_cols)}", cap="From the Kaggle file"), unsafe_allow_html=True)
    b[1].markdown(stat("Added features", f"{len(added_cols)}", cap="Created in this project"), unsafe_allow_html=True)
    b[2].markdown(stat("Final columns", f"{full_df.shape[1]}", cap=f"{len(original_cols)} original + {len(added_cols)} added"), unsafe_allow_html=True)
    st.write("")
    s = st.columns(4)
    steps = [("Step 1", "Quality checks", "Missing cells, duplicate rows and duplicate dates were checked in the raw file."),
             ("Step 2", "Validation", "Dates were parsed and price columns were checked for valid, positive values."),
             ("Step 3", "Project scope", "The analysis focuses on recent gold behaviour, starting from 2020."),
             ("Step 4", "Feature engineering", f"{len(added_cols)} columns were added using only current and past information.")]
    for col, (n, t, d) in zip(s, steps):
        col.markdown(f'<div class="stat"><div class="step-n">{n}</div><div class="step-t">{t}</div><div class="step-d">{d}</div></div>', unsafe_allow_html=True)
    st.write("")
    explain = {"daily_return": ("Daily percentage change in the closing price", "To measure day-to-day movement"),
               "price_range": ("High price minus Low price", "To measure intraday fluctuation"),
               "lag_1": ("Previous day's closing price", "To give the model recent price history"),
               "lag_2": ("Closing price two days earlier", "To capture a longer short-term history"),
               "ma_5": ("5-day moving average of the close", "To capture the short-term trend"),
               "ma_20": ("20-day moving average of the close", "To capture the medium-term trend"),
               "volatility_10": ("Rolling 10-day standard deviation of returns", "To measure short-term risk"),
               "year": ("Year extracted from the date", "To support time-based comparisons"),
               "month": ("Month extracted from the date", "To test seasonality"),
               "day_of_week": ("Weekday number extracted from the date", "To test weekday effects"),
               "weekday": ("Weekday name", "To make categorical analysis readable"),
               "next_close": ("Next trading day's closing price", "Target of the prediction")}
    with panel("Added features", "Every new column, what it means and why it exists"):
        table(pd.DataFrame([{"Added Feature": c, "Meaning": explain.get(c, ("Derived analytical feature", "Added to support EDA or ML"))[0],
                             "Why It Was Added": explain.get(c, ("", "Added to support EDA or machine learning"))[1]} for c in added_cols]), hide_index=True)
        note(f"The final table has {full_df.shape[1]} columns: {len(original_cols)} original columns plus {len(added_cols)} engineered features. None of them uses future information, except the target next_close.")

    # ---------------- C. PROCESSED
    sec(" Processed Data Overview", "The data after cleaning and after feature engineering.")
    c = st.columns(2)
    c[0].markdown(stat("Shape after cleaning", f"{clean_df.shape[0]:,} × {clean_df.shape[1]}", cap="rows × columns", small=True), unsafe_allow_html=True)
    c[1].markdown(stat("Shape after feature engineering", f"{full_df.shape[0]:,} × {full_df.shape[1]}", cap="rows × columns (initial rows without enough history are dropped)", small=True), unsafe_allow_html=True)
    st.write("")
    with panel("Preview after cleaning", "Same columns as the raw file, restricted to the project scope"):
        table(clean_df.head(8))
    with panel("Preview after feature engineering", "Model-ready table with the new columns"):
        pcols = [x for x in ["date", "open", "high", "low", "close", "daily_return", "price_range", "lag_1", "lag_2", "ma_5", "ma_20", "volatility_10", "next_close"] if x in full_df.columns]
        table(full_df[pcols].head(10))
        note("Early rows lose a few days because moving averages and lags need history; this is expected and prevents any look-ahead information.")

    # ---------------- DATA QUALITY
    sec("Data Quality", "Checks performed on the raw file before analysis.")
    miss = raw_df.isnull().sum(); dup_rows = int(raw_df.duplicated().sum())
    dup_dates = int(raw_df.duplicated("date").sum()) if "date" in raw_df.columns else 0
    q = st.columns(3)
    q[0].markdown(stat("Missing cells", f"{int(miss.sum()):,}", cap="All columns together"), unsafe_allow_html=True)
    q[1].markdown(stat("Duplicate rows", f"{dup_rows:,}", cap="Identical full rows"), unsafe_allow_html=True)
    q[2].markdown(stat("Duplicate dates", f"{dup_dates:,}", cap="Same date repeated"), unsafe_allow_html=True)
    st.write("")
    with panel("Missing values by column", "A table is clearer than an empty chart when counts are zero"):
        table(pd.DataFrame({"Column": miss.index, "Missing Values": miss.values, "Missing %": (miss.values / len(raw_df) * 100).round(3)}), hide_index=True)
        note("No cleaning decision was needed for missing values." if miss.sum() == 0 else
             f"{int(miss.sum())} missing cells were found. Columns below 5% missing have their rows dropped; columns at 5% or more are filled with the median.")

    # ---------------- OUTLIERS
    sec("Outlier Analysis", "Review unusual observations across important financial variables.")
    feats = [c for c in ["open", "high", "low", "close", "daily_return", "price_range", "volatility_10"] if c in full_df.columns]
    names = {"open": "Open", "high": "High", "low": "Low", "close": "Close", "daily_return": "Daily Return", "price_range": "Price Range", "volatility_10": "Volatility"}
    with panel("Choose a variable", "The cards and charts below follow your choice"):
        if hasattr(st, "pills"):
            chosen = st.pills("Variable", feats, default=feats[0], format_func=lambda x: names.get(x, x), selection_mode="single", label_visibility="collapsed") or feats[0]
        else:
            chosen = st.radio("Variable", feats, horizontal=True, format_func=lambda x: names.get(x, x), label_visibility="collapsed")
    out = outlier_summary(full_df, chosen)
    series = full_df[chosen].dropna()
    pct_type = chosen in ("daily_return", "volatility_10")
    scale = 100 if (pct_type and series.abs().median() < 0.2) else 1
    unit = "%" if pct_type else "USD"
    f4 = lambda v: f"{v:,.2f}" if abs(v) >= 100 else f"{v:.3f}"
    lo, hi = out["lower_bound"] * scale, out["upper_bound"] * scale
    g = st.columns(4)
    g[0].markdown(ingot_card("Selected Feature", names.get(chosen, chosen), "IQR analysis"), unsafe_allow_html=True)
    g[1].markdown(ingot_card("Lower Bound", f"{f4(lo)}", f"Q1 − 1.5 × IQR · {unit}"), unsafe_allow_html=True)
    g[2].markdown(ingot_card("Upper Bound", f"{f4(hi)}", f"Q3 + 1.5 × IQR · {unit}"), unsafe_allow_html=True)
    g[3].markdown(ingot_card("Outliers", f"{out['count']:,}", f"{out['percent']:.2f}% of rows"), unsafe_allow_html=True)
    st.write("")
    with panel("Spread and distribution", f"{names.get(chosen, chosen)} ({unit}) across the prepared data"):
        l, r = st.columns([1, 1.3])
        sv = series * scale
        with l:
            fig = go.Figure(go.Box(y=sv, boxpoints="outliers", marker_color=GOLD, line_color=GOLD, fillcolor="rgba(212,175,55,.3)", name=names.get(chosen, chosen)))
            fig.add_hline(y=lo, line_dash="dash", line_color=CHAMP); fig.add_hline(y=hi, line_dash="dash", line_color=CHAMP)
            fig.update_layout(yaxis_title=f"{names.get(chosen, chosen)} ({unit})", showlegend=False); plot(fig, 400)
        with r:
            fig = go.Figure(go.Histogram(x=sv, nbinsx=55, marker_color=LIGHT_GOLD, opacity=.9))
            fig.add_vline(x=lo, line_dash="dash", line_color=CHAMP); fig.add_vline(x=hi, line_dash="dash", line_color=CHAMP)
            fig.update_layout(xaxis_title=f"{names.get(chosen, chosen)} ({unit})", yaxis_title="Observations"); plot(fig, 400)
        if chosen in ("open", "high", "low", "close"):
            note(f"IQR flags {out['count']:,} observations ({out['percent']:.2f}%) for {names[chosen]}, but they are kept. Gold trends over time, so a very high or low price is a valid market level rather than a data error.")
        else:
            note(f"IQR flags {out['count']:,} observations ({out['percent']:.2f}%) for {names.get(chosen, chosen)}. They are kept because large returns, ranges and volatility spikes are real market shocks; IQR is used to understand unusual behaviour, not to delete valid financial events.")

# ================================================================== EDA
with tab2:
    sec("Exploratory Data Analysis", "Charts are grouped by the type of relationship they answer.")
    rets = filtered["daily_return"].dropna() * 100

    rel("Univariate analysis", "One Variable at a Time", "Shape, spread and unusual values of individual variables.")
    c1, c2 = st.columns(2)
    with c1:
        with panel("Distribution of daily returns", "How often each daily % change occurs"):
            fig = go.Figure(go.Histogram(x=rets, nbinsx=60, marker_color=GOLD, opacity=.9)); fig.update_layout(xaxis_title="Daily return (%)", yaxis_title="Days"); plot(fig, 340)
            note(f"Returns centre near {rets.mean():.3f}% with a standard deviation of {rets.std():.2f}%. Most days move far less than the long-term trend.")
    with c2:
        with panel("Daily return boxplot", "Median, spread and extreme days"):
            fig = go.Figure(go.Box(y=rets, boxpoints="outliers", marker_color=GOLD, line_color=GOLD, fillcolor="rgba(212,175,55,.3)")); fig.update_layout(yaxis_title="Daily return (%)", showlegend=False); plot(fig, 340)
            q1_, q3_ = rets.quantile(.25), rets.quantile(.75)
            note(f"The middle 50% of days lie between {q1_:.2f}% and {q3_:.2f}%. Points beyond the whiskers are unusually large moves.")
    c1, c2 = st.columns([2, 1])
    with c1:
        with panel("Rolling volatility", "10-day standard deviation of daily returns"):
            vol = filtered.dropna(subset=["volatility_10"])
            fig = go.Figure(area_trace(vol["date"], vol["volatility_10"], "Volatility")); fig.update_layout(yaxis_title="10-day volatility", showlegend=False); plot(fig, 340)
            if len(vol):
                vd = vol.loc[vol["volatility_10"].idxmax(), "date"]
                note(f"The strongest volatility spike occurred around {vd:%d %b %Y}. High-volatility periods make next-day forecasting harder.")
    with c2:
        with panel("Up vs down days", "Share of trading days"):
            up = int((rets > 0).sum()); dn = int((rets <= 0).sum())
            fig = go.Figure(go.Pie(labels=["Up", "Down"], values=[up, dn], hole=.65, marker=dict(colors=[GOLD, "#6B5A24"], line=dict(color="#181612", width=3)), textinfo="percent", textfont=dict(color="#181818", size=15)))
            fig.update_layout(legend=dict(orientation="h", y=-.05)); plot(fig, 340)
            note(f"Gold closed higher on {up / (up + dn) * 100:.1f}% of days.")

    rel("Time-based analysis", "How Gold Changes Over Time", "Long-term price level and yearly behaviour.")
    with panel("Historical gold price", f"Close price with the {ma_choice.split(' ')[0]} moving average"):
        fig = go.Figure(area_trace(filtered["date"], filtered["close"], "Close"))
        fig.add_trace(go.Scatter(x=filtered["date"], y=filtered[ma_col], name=ma_choice.split(" (")[0], line=dict(color="#FFFFFF", width=1.4, dash="dot")))
        fig.update_layout(yaxis_title="USD"); plot(fig, 430)
        tc = (filtered["close"].iloc[-1] / filtered["close"].iloc[0] - 1) * 100
        note(f"Gold moved from {money(filtered['close'].iloc[0])} to {money(filtered['close'].iloc[-1])} ({tc:+.1f}%) over the selected period. The moving average smooths short-term noise and shows the underlying trend.")
    c1, c2 = st.columns(2)
    with c1:
        with panel("Average closing price by year"):
            yearly = filtered.groupby("year", as_index=False)["close"].mean()
            fig = go.Figure(go.Bar(x=yearly["year"].astype(str), y=yearly["close"], marker_color=GOLD)); fig.update_layout(yaxis_title="Average close (USD)"); plot(fig, 350)
            by = yearly.loc[yearly["close"].idxmax()]
            note(f"{int(by['year'])} has the highest average close ({money(by['close'])}).")
    with c2:
        with panel("Return by year (%)", "Last close of each year versus the previous year"):
            yr = (full_df.groupby("year")["close"].last().pct_change() * 100).dropna()
            yr = yr[yr.index.isin(filtered["year"].unique())]
            if len(yr):
                fig = go.Figure(go.Bar(x=yr.index.astype(str), y=yr.values, marker_color=[GOLD if v > 0 else LOSS for v in yr.values])); fig.update_layout(yaxis_title="Return (%)"); plot(fig, 350)
                note(f"Best year: {yr.idxmax()} ({yr.max():+.1f}%). Worst year: {yr.idxmin()} ({yr.min():+.1f}%).")

    rel("Numerical × Numerical", "Relationships Between Numeric Variables", "How price variables and engineered features move together.")
    c1, c2 = st.columns([.9, 1.1])
    with c1:
        with panel("Open vs Close"):
            sc = filtered.dropna(subset=["open", "close"])
            fig = go.Figure(go.Scatter(x=sc["open"], y=sc["close"], mode="markers", marker=dict(color=GOLD, size=6, opacity=.55))); fig.update_layout(xaxis_title="Open price", yaxis_title="Close price"); plot(fig, 420)
            note(f"Open and Close have a correlation of {sc['open'].corr(sc['close']):.3f}. They describe the same day's price level, so the link is almost perfect.")
    with c2:
        with panel("Correlation heatmap"):
            corr = correlation_table(filtered)
            fig = px.imshow(corr, text_auto=".2f", color_continuous_scale=[[0, "#7A3B2E"], [.5, "#2a261d"], [1, "#D4AF37"]], zmin=-1, zmax=1)
            fig.update_traces(textfont=dict(color=WARM, size=12)); fig.update_layout(coloraxis_colorbar=dict(tickfont=dict(color=WARM))); plot(fig, 420)
            note("Open, High, Low and Close move together. Daily return and volatility carry different information, which makes them useful extra features for prediction.")

    rel("Categorical × Numerical", "Calendar Patterns", "Average performance across months and weekdays.")
    c1, c2 = st.columns(2)
    with c1:
        with panel("Month vs average daily return"):
            mo = filtered.groupby("month", as_index=False)["daily_return"].mean(); mo["daily_return"] *= 100; mo["name"] = mo["month"].map(lambda m: MONTHS[int(m) - 1])
            fig = go.Figure(go.Bar(x=mo["name"], y=mo["daily_return"], marker_color=[GOLD if v > 0 else LOSS for v in mo["daily_return"]])); fig.update_layout(yaxis_title="Average daily return (%)"); plot(fig, 350)
            bm, wm = mo.loc[mo["daily_return"].idxmax()], mo.loc[mo["daily_return"].idxmin()]
            note(f"{bm['name']} is the strongest month ({bm['daily_return']:.3f}% per day) and {wm['name']} the weakest ({wm['daily_return']:.3f}%). The gaps are small, so month alone is a weak predictor.")
    with c2:
        with panel("Up vs down days by weekday", "Number of trading days that closed up or down"):
            wk = filtered.assign(dir=np.where(filtered["daily_return"] > 0, "Up", "Down")).groupby(["weekday", "dir"]).size().unstack(fill_value=0)
            wk = wk.reindex([w for w in WEEKDAYS if w in wk.index])
            fig = go.Figure()
            for col_, colr in (("Up", GOLD), ("Down", "#6B5A24")):
                if col_ in wk: fig.add_bar(x=wk.index, y=wk[col_], name=col_, marker_color=colr)
            fig.update_layout(barmode="stack", yaxis_title="Days"); plot(fig, 350)
            wr = (filtered.groupby("weekday")["daily_return"].mean() * 100).sort_values()
            note(f"{wr.index[-1]} has the highest average return ({wr.iloc[-1]:.3f}%) and {wr.index[0]} the lowest ({wr.iloc[0]:.3f}%). Weekday patterns are descriptive, not trading rules.")

# ================================================================== MACHINE LEARNING
with tab3:
    sec("Machine Learning", "Can simple models beat the assumption that tomorrow's price equals today's?")
    m = st.columns(3)
    cards = [("Naive Baseline", "Predicts tomorrow's close as today's close. This is the reference every model must beat."),
             ("Linear Regression", "Learns a simple linear relationship between historical features and the next close."),
             ("Random Forest", "Combines many decision trees to capture non-linear relationships in the features.")]
    for col, (n, d) in zip(m, cards):
        col.markdown(f'<div class="stat"><div class="step-t">{n}</div><div class="step-d">{d}</div></div>', unsafe_allow_html=True)
    st.write("")
    t = st.columns(3)
    t[0].markdown(stat("Training rows", f"{ml['train_rows']:,}", cap="Earlier part of the timeline"), unsafe_allow_html=True)
    t[1].markdown(stat("Testing rows", f"{ml['test_rows']:,}", cap="Later part, never seen in training"), unsafe_allow_html=True)
    t[2].markdown(stat("Split type", "Time-based", cap="No shuffling, no future leakage", small=True), unsafe_allow_html=True)

    metrics = ml["metrics"].copy()
    rel("Model evaluation", "Performance Overview", "Lower MAE and RMSE are better. Higher direction accuracy is better.")
    with panel("Performance table"):
        table(metrics.round(3), hide_index=True)
    bm_ = metrics.loc[metrics["MAE"].idxmin()]; br_ = metrics.loc[metrics["RMSE"].idxmin()]; bd_ = metrics.loc[metrics["Direction Accuracy %"].idxmax()]
    p = st.columns(3)
    p[0].markdown(ingot_card("Best MAE", f"{bm_['MAE']:.2f} USD", bm_["Model"]), unsafe_allow_html=True)
    p[1].markdown(ingot_card("Best RMSE", f"{br_['RMSE']:.2f} USD", br_["Model"]), unsafe_allow_html=True)
    p[2].markdown(ingot_card("Best Direction", f"{bd_['Direction Accuracy %']:.1f}%", bd_["Model"]), unsafe_allow_html=True)
    st.write("")
    naive_mae = float(metrics.loc[metrics["Model"] == "Naive", "MAE"].iloc[0])
    impv = (naive_mae - best_ml_row["MAE"]) / naive_mae * 100

    rel("Metric comparison", "Which Model Wins?", "MAE and RMSE are measured in USD; direction accuracy is the share of correct up/down calls.")
    c = st.columns(3)
    for col, (mname, ttl, lower) in zip(c, (("MAE", "MAE (USD)", True), ("RMSE", "RMSE (USD)", True), ("Direction Accuracy %", "Direction accuracy (%)", False))):
        with col:
            with panel(ttl, "Lower is better" if lower else "Higher is better · 50% = coin flip"):
                fig = go.Figure(go.Bar(x=metrics["Model"], y=metrics[mname], marker_color=[MODEL_COLORS.get(x, GOLD) for x in metrics["Model"]], text=metrics[mname].round(2), textposition="outside", textfont=dict(color=WARM, size=14)))
                if not lower: fig.add_hline(y=50, line_dash="dash", line_color=CHAMP)
                fig.update_layout(showlegend=False, yaxis_title=ttl); plot(fig, 340)
    with panel():
        if impv > 0:
            note(f"{best_ml_row['Model']} improves MAE over the Naive baseline by {impv:.1f}%, so it adds measurable value beyond carrying today's price forward. {bd_['Model']} has the best direction accuracy ({bd_['Direction Accuracy %']:.1f}%).")
        else:
            note(f"{best_ml_row['Model']} is the best ML model but does not beat the Naive baseline on MAE. That is still a meaningful finding: next-day gold prices are very hard to improve on with historical price features alone.")

    pred = ml["predictions"]
    rel("Prediction tracking", "Actual vs Predicted Next Close", "The closer a line is to the white Actual line, the better it follows the market.")
    with panel("Actual vs predicted"):
        fig = go.Figure(go.Scatter(x=pred["date"], y=pred["actual"], name="Actual", line=dict(color="#FFFFFF", width=3)))
        for mn, colr in MODEL_COLORS.items():
            if mn in pred: fig.add_trace(go.Scatter(x=pred["date"], y=pred[mn], name=mn, line=dict(color=colr, width=1.8, dash="dot" if mn == "Naive" else "solid")))
        fig.update_layout(yaxis_title="USD"); plot(fig, 440)
        note("All lines hug the Actual series because tomorrow's price is usually close to today's. What matters is whether the ML models reduce error relative to Naive.")
    with panel("Residuals", "Predicted minus actual, in USD"):
        fig = go.Figure()
        for mn, colr in MODEL_COLORS.items():
            if mn in pred: fig.add_trace(go.Histogram(x=pred[mn] - pred["actual"], name=mn, opacity=.6, marker_color=colr))
        fig.update_layout(barmode="overlay", xaxis_title="Residual (USD)", yaxis_title="Days"); plot(fig, 360)
        parts = [f"{mn}: mean {np.mean(pred[mn] - pred['actual']):+.1f} USD" for mn in MODEL_COLORS if mn in pred]
        note("; ".join(parts) + ". A mean near zero means no systematic over- or under-prediction; the width of each bar group shows the typical error size.")

    rel("Model interpretation", "Random Forest Feature Importance", "Which input variables the forest relied on most.")
    imp = ml["importance"].sort_values("Importance", ascending=False)
    with panel("Feature importance"):
        top10 = imp.head(10).sort_values("Importance")
        fig = go.Figure(go.Bar(x=top10["Importance"], y=top10["Feature"], orientation="h", marker_color=GOLD)); fig.update_layout(xaxis_title="Importance"); plot(fig, 400)
        note(f"The most important feature is {imp.iloc[0]['Feature']} ({imp.iloc[0]['Importance']:.2f}). Importance shows how much the model used a variable; it does not prove causation.")

    rel("Forecast", "Next Trading Day", "Model outputs from the latest prepared observation.")
    f1, f2 = st.columns([1, 2])
    with f1:
        st.markdown(ingot_card("Selected Forecast", money(fc_val), f"{best_ml_row['Model']} · {chg:+.2f}% vs last close"), unsafe_allow_html=True)
    with f2:
        with panel("Forecast by model"):
            ft = fc_tbl.copy(); ft["Forecast"] = ft["Forecast"].round(2); table(ft, hide_index=True)
    with panel():
        note(f"Based on MAE, {best_mae_row['Model']} is the strongest model in this experiment, and the strongest ML model is {best_ml_row['Model']} with a next-close forecast of {money(fc_val)}. This is for educational analysis only and is not financial advice.")