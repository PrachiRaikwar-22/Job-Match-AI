
"""Job Match AI - Streamlit app.  Run:  streamlit run app.py"""
import streamlit as st
from dotenv import load_dotenv

from core import llm
from core.nlp import analyze, highlight_html, retrieve
from core.parser import clean_text, uploaded_to_text

load_dotenv()
# st.set_page_config(page_title="Job Match AI", page_icon="📄", layout="wide")
from PIL import Image
import streamlit as st

# Option A: Use a local logo image (e.g., PNG) for the browser tab icon
tab_icon = Image.open("assets/jobmatch_logo.png")

st.set_page_config(
    page_title="JobMatch AI",
    page_icon=tab_icon,  # Or use an emoji like "🎯" or "📄"
    layout="wide",
)
# Static logo in the main header or sidebar
st.image("assets/jobmatch_logo.png", width=180)

# Or animated logo video (such as your MP4 clip)
with open("assets/jobmatch_logo_animation.mp4", "rb") as video_file:
  st.video(video_file.read(), autoplay=True, loop=True, muted=True)
# ------------------------------------------------------------------ THEME
LIGHT = dict(bg="#ffffff", bg2="#f3f4f8", text="#1f2430", muted="#5b6275",
             border="#d9dce6", accent="#4f46e5", hit="#bbf7d0", hit_t="#14532d",
             miss="#fecaca", miss_t="#7f1d1d")
DARK = dict(bg="#0f1117", bg2="#1a1d27", text="#e8eaf2", muted="#9aa1b5",
            border="#2e3344", accent="#818cf8", hit="#14532d", hit_t="#bbf7d0",
            miss="#7f1d1d", miss_t="#fecaca")


def vars_css(t: dict) -> str:
    return "".join(f"--{k.replace('_', '-')}:{v};" for k, v in t.items())


def inject_theme(mode: str) -> None:
    if mode == "Light":
        root = f":root{{{vars_css(LIGHT)}}}"
    elif mode == "Dark":
        root = f":root{{{vars_css(DARK)}}}"
    else:  # System -> follow the OS setting automatically
        root = (f":root{{{vars_css(LIGHT)}}}"
                f"@media (prefers-color-scheme: dark){{:root{{{vars_css(DARK)}}}}}")
    st.markdown(f"""
<style>
{root}
.stApp, [data-testid="stHeader"] {{ background: var(--bg); color: var(--text); }}
[data-testid="stSidebar"], [data-testid="stSidebar"] > div {{ background: var(--bg2); }}
.stApp p, .stApp label, .stApp span, .stApp h1, .stApp h2, .stApp h3, .stApp li,
.stApp [data-testid="stMarkdownContainer"] {{ color: var(--text); }}
.stTextArea textarea, .stTextInput input, [data-baseweb="select"] > div,
[data-testid="stFileUploaderDropzone"] {{
    background: var(--bg2) !important; color: var(--text) !important;
    border: 1px solid var(--border) !important; }}
.stButton button, .stDownloadButton button {{
    background: var(--accent); color: #fff; border: 0; border-radius: 10px; }}
.stTabs [data-baseweb="tab"] {{ color: var(--muted); }}
.stTabs [aria-selected="true"] {{ color: var(--accent); }}
[data-testid="stMetric"] {{ background: var(--bg2); border: 1px solid var(--border);
    padding: 12px 16px; border-radius: 12px; }}
[data-testid="stMetricValue"], [data-testid="stMetricLabel"] {{ color: var(--text); }}
mark.kw-hit {{ background: var(--hit); color: var(--hit-t); padding: 0 3px; border-radius: 4px; }}
.chip {{ display:inline-block; margin:3px 4px 3px 0; padding:3px 10px; border-radius:999px;
    font-size:.85rem; }}
.chip.ok {{ background: var(--hit); color: var(--hit-t); }}
.chip.no {{ background: var(--miss); color: var(--miss-t); }}
.box {{ background: var(--bg2); border:1px solid var(--border); border-radius:12px;
    padding:14px 18px; max-height:420px; overflow:auto; line-height:1.55; color: var(--text); }}
/* theme switch pinned to the top-right corner */
.st-key-theme_switch {{ position: fixed; top: 10px; right: 70px; z-index: 1000002;
    width: auto !important; }}
.st-key-theme_switch button {{ background: var(--bg2) !important; color: var(--text) !important;
    border: 1px solid var(--border) !important; border-radius: 50% !important;
    width: 42px; height: 42px; padding: 0 !important; font-size: 1.2rem; }}
</style>""", unsafe_allow_html=True)


def current_theme() -> str:
    """'light' or 'dark' - what the user is seeing right now."""
    if st.session_state.get("theme_mode"):
        return st.session_state.theme_mode
    try:  # nothing clicked yet -> follow the browser / OS theme
        return st.context.theme.type or "light"
    except Exception:
        return "light"


def toggle_theme() -> None:
    st.session_state.theme_mode = "dark" if current_theme() == "light" else "light"


with st.container(key="theme_switch"):
    # light now -> show moon (switch to night); dark now -> show sun (switch to day)
    st.button("🌙" if current_theme() == "light" else "☀️", key="theme_btn",
              on_click=toggle_theme, help="Switch day / night mode")

mode = st.session_state.get("theme_mode")  # None = follow system (default)
inject_theme({"light": "Light", "dark": "Dark"}.get(mode, "System"))

TOP_N = 30  # number of keywords to extract

# ------------------------------------------------------------------- MAIN
st.title("📄 Job Match AI")
st.caption("Upload your resume, paste the job description, see how well you match, "
           "then get AI suggestions on what to add or learn.")
c1, c2 = st.columns(2)
with c1:
    st.subheader("1 · Your resume")
    res_file = st.file_uploader("Upload resume (PDF or TXT)", type=["pdf", "txt"], key="res")
with c2:
    st.subheader("2 · Job description")
    jd_paste = st.text_area("Paste JD text", height=250, label_visibility="collapsed",
                            placeholder="Paste the job description here...")

resume_text = uploaded_to_text(res_file)
jd_text = clean_text(jd_paste)

if st.button("🔍 Analyze match", type="primary", use_container_width=True):
    if not resume_text or not jd_text:
        st.warning("Please provide both a resume and a job description.")
    else:
        st.session_state.result = analyze(resume_text, jd_text, TOP_N)
        st.session_state.resume, st.session_state.jd = resume_text, jd_text
        st.session_state.pop("suggestions", None)

res = st.session_state.get("result")
if res:
    r_text, j_text = st.session_state.resume, st.session_state.jd
    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Overall match", f"{res.final_score}%")
    m2.metric("TF-IDF cosine similarity", f"{res.cosine_score}%")
    m3.metric("Keyword coverage", f"{res.keyword_score}%")
    st.progress(min(int(res.final_score), 100))

    k1, k2 = st.columns(2)
    with k1:
        st.markdown(f"#### ✅ Matched ({len(res.matched)})")
        st.markdown("".join(f'<span class="chip ok">{k}</span>' for k in res.matched)
                    or "_none_", unsafe_allow_html=True)
    with k2:
        st.markdown(f"#### ❌ Missing ({len(res.missing)})")
        st.markdown("".join(f'<span class="chip no">{k}</span>' for k in res.missing)
                    or "_none 🎉_", unsafe_allow_html=True)

    t1, t2 = st.tabs(["Resume (matches highlighted)", "Job description"])
    t1.markdown(f'<div class="box">{highlight_html(r_text, res.matched)}</div>',
                unsafe_allow_html=True)
    t2.markdown(f'<div class="box">{highlight_html(j_text, res.matched)}</div>',
                unsafe_allow_html=True)

    # ------------------------------------------------------ SUGGESTIONS
    st.divider()
    st.subheader("3 · Suggestions to improve your match")
    st.caption("AI looks at the missing keywords and checks your resume for related "
               "experience. It never adds skills you don't have.")
    if st.button("💡 Get suggestions", use_container_width=True):
        try:
            with st.spinner("Reading your resume and finding honest improvements..."):
                chunks = retrieve(r_text, j_text, k=6)          # RAG step
                prompt = llm.build_suggestion_prompt(chunks, r_text, j_text,
                                                     res.matched, res.missing)
                st.session_state.suggestions = llm.generate(prompt)
        except Exception as e:
            st.error(f"AI call failed: {e}")

    if st.session_state.get("suggestions"):
        st.markdown(st.session_state.suggestions)
