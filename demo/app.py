"""AI Shield local Streamlit demo -- Self-Supervised Pretraining for Traffic.

Shows the measured property from an actual run of scripts/run_pipeline.py:
fine-tuned accuracy vs label fraction (pretrained vs from-scratch), the
pretraining loss curve, and the t-SNE embedding clusters -- all in the AI
Shield dark theme.

Run:  streamlit run demo/app.py
"""

import base64
import json
import os
import sys

import streamlit as st

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

RESULTS_DIR = os.path.join(REPO_ROOT, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

SUMMARY_PATH = os.path.join(RESULTS_DIR, "summary.json")
FIG_ACC = os.path.join(FIGURES_DIR, "label_fraction_accuracy.png")
FIG_LOSS = os.path.join(FIGURES_DIR, "pretraining_loss.png")
FIG_TSNE = os.path.join(FIGURES_DIR, "tsne_clusters.png")

st.set_page_config(page_title="AI Shield | Self-Supervised Traffic", page_icon="\U0001F6A9", layout="wide")

AI_SHIELD_CSS = """
<style>
  :root {
    --bg: #030303; --card: #09090b; --card-95: rgba(9,9,11,.95);
    --border: #18181b; --border-strong: #27272a;
    --emerald-400: #34d399; --emerald-500: #10b981;
    --purple-400: #c084fc; --cyan-500: #06b6d2;
    --red-400: #f87171; --amber-400: #fbbf24;
    --text: #ffffff; --text-2: #d4d4d8; --text-3: #a1a1aa; --text-4: #71717a; --text-5: #52525b;
    --font-heading: "Space Grotesk", "Plus Jakarta Sans", sans-serif;
    --font-mono: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
  }
  .stApp { background: var(--bg); }
  .block-container { padding-top: 1.6rem; max-width: 1100px; }
  h1,h2,h3,h4 { font-family: var(--font-heading); color: var(--text); letter-spacing: -0.02em; }
  .stMarkdown p, .stMarkdown li { color: var(--text-2); }
  .grad-title { font-family: var(--font-heading); font-size: 2.1rem; font-weight: 700; line-height: 1.15;
    background: linear-gradient(135deg,#fff,#f4f4f5 50%,#a1a1aa);
    -webkit-background-clip: text; background-clip: text; color: transparent; }
  .grad-accent { background: linear-gradient(90deg,#34d399,#5eead4,#06b6d2);
    -webkit-background-clip: text; background-clip: text; color: transparent; }
  .section-label { font-family: var(--font-mono); font-size: 11px; color: var(--emerald-400);
    letter-spacing: .15em; text-transform: uppercase; }
  .card { border-radius: 16px; border: 1px solid var(--border); background: var(--card-95);
    box-shadow: 0 20px 50px rgba(0,0,0,.85); padding: 20px; margin-bottom: 14px; }
  .stat-card { padding: 18px; border-radius: 16px; border: 1px solid var(--border);
    background: var(--card-95); box-shadow: 0 8px 30px rgba(0,0,0,.8); }
  .stat-card .label { font-family: var(--font-mono); font-size: 10px; color: var(--text-4);
    letter-spacing: .12em; text-transform: uppercase; margin-bottom: 6px; }
  .stat-card .value { font-family: var(--font-mono); font-size: 22px; font-weight: 800; color: var(--text); }
  .stat-card .sub { font-family: var(--font-mono); font-size: 11px; color: var(--text-4); margin-top: 6px; }
  .framed { width: 100%; border-radius: 14px; border: 1px solid var(--border-strong); }
  .badge { font-family: var(--font-mono); font-size: 9px; font-weight: 700; text-transform: uppercase;
    padding: 3px 8px; border-radius: 4px; background: rgba(16,185,129,.1); color: var(--emerald-400);
    letter-spacing: .08em; display: inline-block; margin-right: 6px; }
  .badge.danger { background: rgba(239,68,68,.1); color: var(--red-400); }
  .badge.purple { background: rgba(192,132,252,.1); color: var(--purple-400); }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th { font-family: var(--font-mono); font-size: 10px; letter-spacing: .1em; text-transform: uppercase;
    color: var(--text-4); text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--border-strong); }
  td { padding: 8px 10px; border-bottom: 1px solid var(--border); color: var(--text-2); }
  .footer { border-top: 1px solid rgba(255,255,255,.05); padding-top: 1.4rem; margin-top: 2rem;
    text-align: center; color: var(--text-4); font-size: 13px; }
  .mono { font-family: var(--font-mono); }
  a { color: var(--emerald-400); }
</style>
"""
st.markdown(AI_SHIELD_CSS, unsafe_allow_html=True)


def _img_html(path: str, caption: str) -> str:
    with open(path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode("ascii")
    return (
        f'<div class="card"><img class="framed" src="data:image/png;base64,{b64}" alt="{caption}"/>'
        f'<p class="mono" style="font-size:11px;color:#71717a;margin:10px 0 0;letter-spacing:.08em;text-transform:uppercase;">'
        f"{caption}</p></div>"
    )


def _load_summary():
    if os.path.exists(SUMMARY_PATH):
        with open(SUMMARY_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    return None


def _stat_card(label, value, sub=""):
    return f'<div class="stat-card"><div class="label">{label}</div><div class="value">{value}</div><div class="sub">{sub}</div></div>'


summary = _load_summary()

# ---- Nav ----
st.markdown(
    """
    <div style="display:flex;align-items:center;justify-content:space-between;
        border-bottom:1px solid rgba(255,255,255,.05); padding-bottom:.8rem; margin-bottom:1.6rem;">
      <div style="display:flex;align-items:center;gap:10px;font-family:'Space Grotesk',sans-serif;font-weight:700;color:#fff;">
        <span style="width:10px;height:10px;border-radius:50%;background:#34d399;box-shadow:0 0 10px #10b981;"></span>
        AI SHIELD &nbsp;--&nbsp; Self-Supervised Traffic Pretraining
      </div>
      <a href="https://github.com/Phanidhar007/self-supervised-traffic"
         class="mono" style="color:#a1a1aa;text-decoration:none;font-size:12px;
         border:1px solid #27272a;border-radius:9999px;padding:6px 14px;">GitHub</a>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---- Hero ----
st.markdown(
    '<span class="section-label">Pretrain &middot; Fine-tune &middot; Compare</span>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="grad-title">Pretraining beats from-scratch<br>'
    'when labels are <span class="grad-accent">scarce</span>.</div>',
    unsafe_allow_html=True,
)
st.markdown(
    """
    <p>We pretrain a small transformer encoder on a <b>large unlabeled</b> pool of network flows
    using a <span class="mono">masked-feature reconstruction</span> objective, then fine-tune it on a
    <b>tiny labeled</b> IDS set and compare against training the same encoder <b>from scratch</b>.
    The gap widens as labels shrink -- the key selling point for security teams with little labeled data.
    </p>
    """,
    unsafe_allow_html=True,
)

# ---- Stat cards ----
if summary is not None:
    s = summary
    sw = s["sweep"]
    frac0 = sw["fractions"][0]
    pre0, scr0 = sw["pretrained"][0], sw["scratch"][0]
    gain = (pre0 - scr0) * 100
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            _stat_card("Gain at %.0f%% labels" % (frac0 * 100), f"+{gain:.1f} pts",
                       f"pretrained {pre0*100:.1f}% vs scratch {scr0*100:.1f}%"),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            _stat_card("Unlabeled pool", f"{s['dataset']['n_unlabeled']:,} flows",
                       "used for masked-feature pretraining"),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            _stat_card("Pretrain final MSE", f"{s['pretrain']['final_loss']:.4f}",
                       f"{s['pretrain']['epochs']} epochs, mask {s['pretrain']['mask_frac']*100:.0f}%"),
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            _stat_card("Test set", f"{s['dataset']['n_test']:,} flows",
                       "held out, unseen in pretraining"),
            unsafe_allow_html=True,
        )

    # ---- Comparison chart ----
    st.markdown('<span class="section-label">1 &middot; Low-label accuracy</span>', unsafe_allow_html=True)
    st.markdown(
        _img_html(os.path.join(FIGURES_DIR, "label_fraction_accuracy.png"),
                  "fine-tuned accuracy vs label fraction -- pretrained vs from-scratch"),
        unsafe_allow_html=True,
    )
    st.markdown(
        f"> **Low-label highlight:** with only **{sw['n_train'][0]} labeled flows** "
        f"(**{frac0*100:.0f}%** of the labeled set), the pretrained encoder reaches "
        f"**{pre0*100:.1f}%** accuracy vs **{scr0*100:.1f}%** from scratch "
        f"(`+{gain:.1f}` points). Unlabeled structure is transferred, so labels matter less.",
        unsafe_allow_html=True,
    )

    # ---- Table ----
    rows = "".join(
        f"<tr><td>{f*100:.0f}%</td><td>{nt}</td>"
        f"<td style='color:#34d399;font-weight:700;'>{p*100:.2f}%</td>"
        f"<td>{sc*100:.2f}%</td>"
        f"<td style='color:#34d399;'>+{(p-sc)*100:.2f} pts</td></tr>"
        for f, nt, p, sc in zip(sw["fractions"], sw["n_train"], sw["pretrained"], sw["scratch"])
    )
    st.markdown(
        '<div class="card"><table><thead><tr>'
        "<th>Label fraction</th><th>Flows used</th><th>Pretrained</th><th>From scratch</th><th>Gain</th>"
        "</tr></thead><tbody>" + rows + "</tbody></table></div>",
        unsafe_allow_html=True,
    )

    # ---- Training curves + t-SNE ----
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown('<span class="section-label">2 &middot; Pretraining loss</span>', unsafe_allow_html=True)
        st.markdown(
            _img_html(os.path.join(FIGURES_DIR, "pretraining_loss.png"), "masked-feature reconstruction MSE per epoch"),
            unsafe_allow_html=True,
        )
    with col_r:
        st.markdown('<span class="section-label">3 &middot; Embedding clusters</span>', unsafe_allow_html=True)
        st.markdown(
            _img_html(os.path.join(FIGURES_DIR, "tsne_clusters.png"), "t-SNE of encoder embeddings, colored by archetype"),
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="card">
          <span class="badge purple">normal</span><span class="badge danger">attack</span>
          <span class="mono" style="font-size:12px;color:#a1a1aa;">
          The pretrained encoder discovers traffic structure -- web, dns, ssh, mail, video
          and attack flows (portscan, bruteforce, exfil) separate into visible clusters even though
          pretraining never saw a label.
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.warning("No results/summary.json found. Run `python scripts/run_pipeline.py` first to generate the figures and metrics.")

st.markdown(
    '<div class="footer">AI SHIELD &mdash; self-supervised traffic pretraining &middot; '
    'masked-feature reconstruction &middot; run pipeline to regenerate metrics</div>',
    unsafe_allow_html=True,
)
