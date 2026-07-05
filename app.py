"""
ForestGuard — Satellite Deforestation Detection Web App
A Streamlit application wrapping the deforestation detection ML pipeline.

Run with:  streamlit run app.py
"""

import sys
import os
import random

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from PIL import Image

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from detect_deforestation import (
    split_image_into_patches,
    classify_patches,
    build_land_cover_map,
    detect_changes,
    CLASS_NAMES,
    CLASS_COLORS,
    DEFORESTATION_TARGETS,
    DEVICE,
    OUTPUT_DIR,
    DATA_DIR,
    MODEL_PATH,
    NUM_CLASSES,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ForestGuard — Satellite Deforestation Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS injection ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

.hero {
    background: linear-gradient(135deg, #0a1f14 0%, #0f3d23 55%, #071a0f 100%);
    border: 1px solid #1a4a2e; border-radius: 20px;
    padding: 44px 48px; margin-bottom: 32px;
    position: relative; overflow: hidden;
}
.hero::before {
    content: ''; position: absolute; top: -80px; right: -80px;
    width: 380px; height: 380px;
    background: radial-gradient(circle, rgba(63,185,80,0.10) 0%, transparent 70%);
    pointer-events: none;
}
.hero-badge {
    display: inline-block;
    background: rgba(63,185,80,0.15); border: 1px solid rgba(63,185,80,0.35);
    color: #3fb950; padding: 5px 14px; border-radius: 20px;
    font-size: 0.72rem; font-weight: 700; margin-bottom: 18px;
    letter-spacing: 1.2px; text-transform: uppercase;
}
.hero-title {
    font-size: 2.8rem; font-weight: 800; color: #ffffff;
    margin: 0 0 10px 0; letter-spacing: -1px; line-height: 1.1;
}
.hero-title span { color: #3fb950; }
.hero-subtitle { color: #8b949e; font-size: 1rem; line-height: 1.6; max-width: 560px; }
.hero-pill {
    display: inline-block; background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12); color: #c9d1d9;
    padding: 4px 12px; border-radius: 20px; font-size: 0.78rem; margin: 14px 4px 0 0;
}

.stat-card {
    background: #161b22; border: 1px solid #21262d; border-radius: 14px;
    padding: 22px 18px; text-align: center;
    transition: border-color 0.25s, transform 0.2s;
}
.stat-card:hover { border-color: #3fb950; transform: translateY(-2px); }
.stat-value { font-size: 2.1rem; font-weight: 700; color: #3fb950; line-height: 1; margin-bottom: 6px; }
.stat-label { font-size: 0.75rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.6px; }

.sec-hdr {
    font-size: 1.05rem; font-weight: 600; color: #e6edf3;
    padding-bottom: 10px; border-bottom: 1px solid #21262d; margin: 24px 0 16px 0;
}

.info-box {
    background: rgba(56,139,253,0.08); border: 1px solid rgba(56,139,253,0.25);
    border-radius: 10px; padding: 14px 18px; color: #79c0ff; font-size: 0.88rem; line-height: 1.5;
}
.warn-box {
    background: rgba(248,81,73,0.08); border: 1px solid rgba(248,81,73,0.25);
    border-radius: 10px; padding: 14px 18px; color: #ffa198; font-size: 0.88rem; line-height: 1.5;
}
.ok-box {
    background: rgba(63,185,80,0.08); border: 1px solid rgba(63,185,80,0.25);
    border-radius: 10px; padding: 14px 18px; color: #56d364; font-size: 0.88rem; line-height: 1.5;
}

.model-card { background: #0d1117; border: 1px solid #21262d; border-radius: 12px; padding: 14px 16px; margin: 8px 0; }
.mc-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 5px 0; border-bottom: 1px solid #1c2128; font-size: 0.82rem;
}
.mc-row:last-child { border-bottom: none; }
.mc-k { color: #8b949e; }
.mc-v { color: #e6edf3; font-weight: 500; }

.trans-row { display: flex; align-items: center; padding: 7px 0; border-bottom: 1px solid #1c2128; gap: 12px; }
.trans-bar-bg { flex: 1; height: 7px; background: #21262d; border-radius: 4px; overflow: hidden; }
.trans-bar-fill { height: 100%; background: linear-gradient(90deg, #f85149 0%, #ff8080 100%); border-radius: 4px; }

.empty-state { text-align: center; padding: 80px 24px; color: #8b949e; }
.empty-icon { font-size: 3.5rem; margin-bottom: 16px; }
.empty-title { font-size: 1.25rem; font-weight: 600; color: #c9d1d9; margin-bottom: 8px; }

div.stButton > button {
    background: linear-gradient(135deg, #238636 0%, #2ea043 100%) !important;
    color: #ffffff !important; border: none !important; border-radius: 8px !important;
    font-weight: 600 !important; font-family: 'Inter', sans-serif !important;
    transition: all 0.2s !important; box-shadow: 0 2px 8px rgba(35,134,54,0.25);
}
div.stButton > button:hover {
    background: linear-gradient(135deg, #2ea043 0%, #3fb950 100%) !important;
    transform: translateY(-2px) !important; box-shadow: 0 6px 20px rgba(63,185,80,0.35) !important;
}
div.stButton > button:disabled { background: #21262d !important; color: #484f58 !important; box-shadow: none !important; }
div.stDownloadButton > button {
    background: #161b22 !important; border: 1px solid #30363d !important;
    color: #c9d1d9 !important; border-radius: 8px !important;
}
div.stDownloadButton > button:hover { border-color: #3fb950 !important; color: #3fb950 !important; }
</style>
""", unsafe_allow_html=True)


# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_model():
    if not os.path.exists(MODEL_PATH):
        return None, None, f"Model not found at `{MODEL_PATH}`. Run `python src/train.py` first."
    try:
        import torch.nn as nn
        from torchvision import models as tv_models
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        model = tv_models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(DEVICE)
        model.eval()
        return model, checkpoint, None
    except Exception as e:
        return None, None, str(e)


# ── Pipeline ──────────────────────────────────────────────────────────────────
def run_pipeline(model, before_path, after_path, patch_size):
    prog = st.progress(0, text="Splitting 'Before' image into patches...")
    before_patches, before_grid, _ = split_image_into_patches(before_path, patch_size)
    prog.progress(15, text="Classifying 'Before' patches... (this may take a moment)")
    before_results = classify_patches(model, before_patches)
    map_before, _ = build_land_cover_map(before_results, before_grid, patch_size)
    prog.progress(50, text="Splitting 'After' image into patches...")
    after_patches, after_grid, _ = split_image_into_patches(after_path, patch_size)
    prog.progress(60, text="Classifying 'After' patches...")
    after_results = classify_patches(model, after_patches)
    map_after, _ = build_land_cover_map(after_results, after_grid, patch_size)
    prog.progress(90, text="Detecting deforestation events...")
    if before_grid != after_grid:
        min_rows = min(map_before.shape[0], map_after.shape[0])
        min_cols = min(map_before.shape[1], map_after.shape[1])
        map_before = map_before[:min_rows, :min_cols]
        map_after = map_after[:min_rows, :min_cols]
    changes, stats = detect_changes(map_before, map_after)
    prog.progress(100, text="Done!")
    prog.empty()
    return map_before, map_after, changes, stats


def build_demo_scenes(patch_size):
    def load_class_images(cls, count=20):
        d = os.path.join(DATA_DIR, cls)
        if not os.path.exists(d):
            return []
        imgs = []
        for fname in sorted(os.listdir(d))[:count]:
            try:
                imgs.append(Image.open(os.path.join(d, fname)).convert("RGB").resize((patch_size, patch_size)))
            except Exception:
                pass
        return imgs

    grid_cols, grid_rows = 10, 8
    forest   = load_class_images("Forest", 80)
    crop     = load_class_images("AnnualCrop", 20)
    resid    = load_class_images("Residential", 20)
    pasture  = load_class_images("Pasture", 20)
    river    = load_class_images("River", 10)
    highway  = load_class_images("Highway", 10)

    if not forest:
        st.error("Cannot find EuroSAT Forest images at `data/EuroSAT/2750/Forest/`.")
        st.stop()

    random.seed(42)
    before_scene = Image.new("RGB", (grid_cols * patch_size, grid_rows * patch_size))
    before_layout = []
    for r in range(grid_rows):
        row = []
        for c in range(grid_cols):
            rnd = random.random()
            if rnd < 0.70:
                img = random.choice(forest); row.append("Forest")
            elif rnd < 0.80:
                img = random.choice(pasture) if pasture else random.choice(forest); row.append("Pasture")
            elif rnd < 0.85:
                img = random.choice(river) if river else random.choice(forest); row.append("River")
            elif rnd < 0.90:
                img = random.choice(highway) if highway else random.choice(forest); row.append("Highway")
            else:
                img = random.choice(crop) if crop else random.choice(forest); row.append("AnnualCrop")
            before_scene.paste(img, (c * patch_size, r * patch_size))
        before_layout.append(row)

    after_scene = Image.new("RGB", (grid_cols * patch_size, grid_rows * patch_size))
    zones = set()
    for r in range(2, 5):
        for c in range(3, 7):
            zones.add((r, c))
    for r, c in [(1, 1), (1, 2), (6, 5), (6, 6), (7, 8)]:
        zones.add((r, c))

    for r in range(grid_rows):
        for c in range(grid_cols):
            if (r, c) in zones and before_layout[r][c] == "Forest":
                img = (random.choice(crop) if crop else random.choice(forest)) if (2 <= r <= 4 and 3 <= c <= 6) \
                      else (random.choice(resid) if resid else random.choice(forest))
            else:
                x, y = c * patch_size, r * patch_size
                img = before_scene.crop((x, y, x + patch_size, y + patch_size))
            after_scene.paste(img, (c * patch_size, r * patch_size))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    bp = os.path.join(OUTPUT_DIR, "app_demo_before.png")
    ap = os.path.join(OUTPUT_DIR, "app_demo_after.png")
    before_scene.save(bp); after_scene.save(ap)
    return bp, ap, before_scene, after_scene


# ── Figures ───────────────────────────────────────────────────────────────────
def _to_arr(lcm, patch_size):
    rows, cols = lcm.shape
    arr = np.zeros((rows * patch_size, cols * patch_size, 3), dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            color = CLASS_COLORS.get(lcm[r, c], (200, 200, 200))
            arr[r*patch_size:(r+1)*patch_size, c*patch_size:(c+1)*patch_size] = color
    return arr


def make_landcover_fig(lcm, patch_size, title):
    arr = _to_arr(lcm, patch_size)
    present = set(lcm.flatten())
    fig, ax = plt.subplots(figsize=(8, 6), facecolor="#0d1117")
    ax.imshow(arr); ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=10)
    ax.axis("off"); ax.set_facecolor("#0d1117")
    patches = [mpatches.Patch(color=np.array(c)/255, label=n)
               for n, c in CLASS_COLORS.items() if n in present]
    if patches:
        ax.legend(handles=patches, loc="lower right", fontsize=7, framealpha=0.9,
                  facecolor="#161b22", edgecolor="#30363d", labelcolor="white",
                  title="Land Cover", title_fontsize=7)
    fig.tight_layout(pad=0.5)
    return fig


def make_deforestation_fig(mb, ma, changes, patch_size):
    rows, cols = mb.shape
    ba = _to_arr(mb, patch_size); aa = _to_arr(ma, patch_size)
    ca = (aa * 0.3).astype(np.uint8)
    for ch in changes:
        r, c = ch["row"], ch["col"]
        ca[r*patch_size:(r+1)*patch_size, c*patch_size:(c+1)*patch_size] = [255, 50, 50]
    for r in range(rows):
        for c in range(cols):
            if mb[r, c] == "Forest" and ma[r, c] == "Forest":
                ca[r*patch_size:(r+1)*patch_size, c*patch_size:(c+1)*patch_size] = [34, 139, 34]
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), facecolor="#0d1117")
    for ax, (t, arr) in zip(axes, [("Before (Time Period 1)", ba), ("After (Time Period 2)", aa),
                                    (f"Deforestation — {len(changes)} events", ca)]):
        ax.imshow(arr); ax.set_title(t, color="white", fontsize=11, fontweight="bold")
        ax.axis("off"); ax.set_facecolor("#0d1117")
    axes[2].legend(handles=[mpatches.Patch(facecolor="#ff3232", label="Deforestation"),
                             mpatches.Patch(facecolor="#228b22", label="Remaining Forest"),
                             mpatches.Patch(facecolor="#555555", label="Other Land Cover")],
                   loc="lower right", fontsize=9, framealpha=0.9,
                   facecolor="#161b22", edgecolor="#30363d", labelcolor="white")
    fig.suptitle("Satellite-Based Deforestation Detection", color="white",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout(pad=0.5)
    return fig


def make_pie(tc):
    labels = list(tc.keys()); sizes = [tc[l] for l in labels]
    colors = ["#{:02x}{:02x}{:02x}".format(*CLASS_COLORS.get(l, (200, 200, 200))) for l in labels]
    fig, ax = plt.subplots(figsize=(5, 4), facecolor="#0d1117")
    ax.pie(sizes, labels=labels, colors=colors, autopct="%1.0f%%", startangle=140,
           wedgeprops={"linewidth": 2, "edgecolor": "#0d1117"},
           textprops={"color": "white", "fontsize": 9})
    ax.set_title("Deforestation Type Breakdown", color="white", fontsize=10, fontweight="bold")
    fig.tight_layout()
    return fig


def make_report(changes, stats, mb, ma):
    lines = ["Satellite-Based Deforestation Detection Report", "=" * 60,
             f"Total patches analyzed : {stats['total_patches']}",
             f"Forest patches (before): {stats['forest_before']}",
             f"Forest patches (after) : {stats['forest_after']}",
             f"Net forest change      : {stats['forest_lost']:+d}",
             f"Deforestation events   : {stats['deforestation_events']}"]
    if stats["forest_before"] > 0:
        lines.append(f"Deforestation rate     : {stats['deforestation_rate']:.1%}")
    lines.append("")
    if changes:
        lines.append("Transition Breakdown:")
        tc = {}
        for ch in changes: tc[ch["to_class"]] = tc.get(ch["to_class"], 0) + 1
        for cls, cnt in sorted(tc.items(), key=lambda x: -x[1]):
            lines.append(f"  Forest -> {cls}: {cnt}")
        lines += ["", "All Events:"]
        for i, ch in enumerate(changes):
            lines.append(f"  [{i+1:>3}] Row={ch['row']}, Col={ch['col']}: {ch['from_class']} -> {ch['to_class']}")
    else:
        lines.append("No deforestation events detected.")
    lines += ["", "Land Cover Distribution:",
              f"  {'Class':<24} {'Before':>8} {'After':>8} {'Change':>8}", "  " + "-" * 50]
    for cls in CLASS_NAMES:
        bc = int(np.sum(mb == cls)); ac = int(np.sum(ma == cls)); ch = ac - bc
        lines.append(f"  {cls:<24} {bc:>8} {ac:>8} {f'{ch:+d}' if ch else '—':>8}")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🌿 ForestGuard")
    st.markdown('<div style="color:#8b949e;font-size:0.8rem;margin-top:-8px;margin-bottom:16px;">Satellite Deforestation Detection</div>', unsafe_allow_html=True)

    with st.spinner("Loading model..."):
        model, checkpoint, model_error = get_model()

    if model_error:
        st.markdown(f'<div class="warn-box">⚠️ {model_error}</div>', unsafe_allow_html=True)
    else:
        val_acc = checkpoint.get("val_accuracy", 0.0)
        epoch   = checkpoint.get("epoch", "?")
        dev_lbl = "🚀 GPU (CUDA)" if "cuda" in str(DEVICE) else "💻 CPU"
        st.markdown(f"""<div class="model-card">
          <div style="color:#3fb950;font-weight:600;font-size:0.82rem;margin-bottom:8px;">● Model Ready</div>
          <div class="mc-row"><span class="mc-k">Architecture</span><span class="mc-v">ResNet-50</span></div>
          <div class="mc-row"><span class="mc-k">Val Accuracy</span><span class="mc-v" style="color:#3fb950;">{val_acc:.2%}</span></div>
          <div class="mc-row"><span class="mc-k">Trained Epochs</span><span class="mc-v">{epoch}</span></div>
          <div class="mc-row"><span class="mc-k">Device</span><span class="mc-v">{dev_lbl}</span></div>
          <div class="mc-row"><span class="mc-k">Classes</span><span class="mc-v">{len(CLASS_NAMES)}</span></div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div style="font-size:0.82rem;font-weight:600;color:#c9d1d9;margin-bottom:8px;">Land Cover Classes</div>', unsafe_allow_html=True)
    for name, color in CLASS_COLORS.items():
        hx = "#{:02x}{:02x}{:02x}".format(*color)
        badge = " 🔴" if name in DEFORESTATION_TARGETS else ""
        st.markdown(f'<div style="font-size:0.8rem;padding:2px 0;color:#c9d1d9;"><span style="color:{hx};font-size:1rem;">■</span> {name}{badge}</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.72rem;color:#8b949e;margin-top:8px;">🔴 = Deforestation target</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div style="font-size:0.72rem;color:#484f58;text-align:center;">ForestGuard v1.0 · ResNet50 + EuroSAT</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# HERO
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <div class="hero-badge">🛰️ AI-Powered Remote Sensing</div>
  <div class="hero-title">Forest<span>Guard</span></div>
  <div class="hero-subtitle">
    Detect deforestation from satellite imagery using deep learning.<br>
    Upload <em>before</em> &amp; <em>after</em> images to analyze land cover change — or run the built-in demo.
  </div>
  <div>
    <span class="hero-pill">🤖 ResNet-50</span>
    <span class="hero-pill">📡 EuroSAT Dataset</span>
    <span class="hero-pill">🌲 10 Land Cover Classes</span>
    <span class="hero-pill">🔍 Patch-Based Detection</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════
tab_analyze, tab_results, tab_model_tab = st.tabs(["🛰️  Analyze", "📊  Results", "📈  Model Metrics"])


# ───────────────────────────────────────────────────────────────
# TAB 1 — ANALYZE
# ───────────────────────────────────────────────────────────────
with tab_analyze:
    if model is None:
        st.markdown('<div class="warn-box">⚠️ No trained model found. Run <code>python src/train.py</code> to train it first, then restart this app.</div>', unsafe_allow_html=True)
        st.stop()

    st.markdown('<div class="sec-hdr">⚙️ Select Analysis Mode</div>', unsafe_allow_html=True)
    mode = st.radio("Mode", options=[
        "🎬  Demo Mode  (uses EuroSAT sample images — no upload needed)",
        "📁  Upload My Own Images"
    ], label_visibility="collapsed")
    st.markdown("---")

    # ── DEMO MODE ────────────────────────────────────────────────────────────
    if "Demo" in mode:
        st.markdown('<div class="info-box"><strong>Demo Mode</strong> — Builds two synthetic satellite scenes from EuroSAT images: a forested "Before" scene and an "After" scene with some forest cleared. The full detection pipeline runs on both. No data upload required.</div>', unsafe_allow_html=True)
        st.markdown("")
        col_ps, _ = st.columns([1, 3])
        with col_ps:
            patch_size = st.select_slider("Patch size (px)", options=[32, 64, 128], value=64,
                                          help="Smaller = finer spatial resolution but slower inference.")
        st.markdown("")

        if st.button("▶  Run Demo Analysis"):
            with st.status("Running demo analysis...", expanded=True) as status:
                st.write("🌿 Building synthetic satellite scenes from EuroSAT images...")
                before_path, after_path, before_pil, after_pil = build_demo_scenes(patch_size)
                st.write("🔍 Running classification pipeline...")
                map_before, map_after, changes, stats = run_pipeline(model, before_path, after_path, patch_size)
                st.write("✅ Analysis complete!")
                status.update(label="Analysis complete!", state="complete")

            st.session_state["results"] = {
                "map_before": map_before, "map_after": map_after,
                "changes": changes, "stats": stats, "patch_size": patch_size,
                "before_img": before_pil, "after_img": after_pil,
            }
            st.markdown('<div class="ok-box">✅ Analysis complete! Switch to the <strong>📊 Results</strong> tab to view the outputs.</div>', unsafe_allow_html=True)

    # ── UPLOAD MODE ──────────────────────────────────────────────────────────
    else:
        st.markdown('<div class="info-box"><strong>Upload Mode</strong> — Upload two satellite images of the <em>same area</em> from different time periods. Supported: PNG, JPG, TIF/TIFF.</div>', unsafe_allow_html=True)
        st.markdown("")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**📅 Before Image** &nbsp;(earlier time period)")
            before_file = st.file_uploader("Before", type=["png", "jpg", "jpeg", "tif", "tiff"], key="up_before", label_visibility="collapsed")
            if before_file:
                st.image(Image.open(before_file).convert("RGB"), caption=f"📸 {before_file.name}", use_container_width=True)
        with col2:
            st.markdown("**📅 After Image** &nbsp;(later time period)")
            after_file = st.file_uploader("After", type=["png", "jpg", "jpeg", "tif", "tiff"], key="up_after", label_visibility="collapsed")
            if after_file:
                st.image(Image.open(after_file).convert("RGB"), caption=f"📸 {after_file.name}", use_container_width=True)

        st.markdown("")
        col_ps2, _ = st.columns([1, 3])
        with col_ps2:
            patch_size = st.select_slider("Patch size (px)", options=[32, 64, 128], value=64, key="ps_upload",
                                          help="Smaller = finer detail but slower inference.")
        st.markdown("")

        both = before_file is not None and after_file is not None
        if not both:
            st.markdown('<div class="info-box">⬆️ Upload both images above to enable analysis.</div>', unsafe_allow_html=True)

        if st.button("▶  Run Analysis", disabled=not both):
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            bp = os.path.join(OUTPUT_DIR, "app_upload_before.png")
            ap = os.path.join(OUTPUT_DIR, "app_upload_after.png")
            before_file.seek(0); after_file.seek(0)
            before_pil = Image.open(before_file).convert("RGB")
            after_pil  = Image.open(after_file).convert("RGB")
            before_pil.save(bp); after_pil.save(ap)

            with st.status("Running deforestation detection...", expanded=True) as status:
                st.write("🔍 Running classification pipeline...")
                map_before, map_after, changes, stats = run_pipeline(model, bp, ap, patch_size)
                st.write("✅ Analysis complete!")
                status.update(label="Analysis complete!", state="complete")

            st.session_state["results"] = {
                "map_before": map_before, "map_after": map_after,
                "changes": changes, "stats": stats, "patch_size": patch_size,
                "before_img": before_pil, "after_img": after_pil,
            }
            st.markdown('<div class="ok-box">✅ Analysis complete! Switch to the <strong>📊 Results</strong> tab to view the outputs.</div>', unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────────
# TAB 2 — RESULTS
# ───────────────────────────────────────────────────────────────
with tab_results:
    if "results" not in st.session_state:
        st.markdown("""<div class="empty-state">
            <div class="empty-icon">🛰️</div>
            <div class="empty-title">No Analysis Yet</div>
            <div>Run an analysis in the <strong>🛰️ Analyze</strong> tab to see results here.</div>
        </div>""", unsafe_allow_html=True)
    else:
        res = st.session_state["results"]
        mb = res["map_before"]; ma = res["map_after"]
        changes = res["changes"]; stats = res["stats"]; ps = res["patch_size"]
        dp = stats["deforestation_rate"] * 100

        # Stat cards
        st.markdown('<div class="sec-hdr">📊 Detection Summary</div>', unsafe_allow_html=True)
        sev = ('<span style="color:#f85149;">🔴 High</span>' if dp > 20 else
               '<span style="color:#e3b341;">🟡 Moderate</span>' if dp > 5 else
               '<span style="color:#3fb950;">🟢 Low</span>')
        lc = "#f85149" if stats["forest_lost"] > 0 else "#3fb950"

        cols = st.columns(5)
        for col, (val, color, lbl) in zip(cols, [
            (str(stats["total_patches"]), "#c9d1d9", "Total Patches"),
            (str(stats["forest_before"]), "#3fb950", "Forest Before"),
            (str(stats["forest_after"]),  "#56d364", "Forest After"),
            (f"{stats['forest_lost']:+d}", lc,       "Net Change"),
            (f"{dp:.1f}%", "#f85149",                f"Deforestation Rate<br><small>{sev}</small>"),
        ]):
            with col:
                st.markdown(f'<div class="stat-card"><div class="stat-value" style="color:{color};">{val}</div><div class="stat-label">{lbl}</div></div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # Input images
        if "before_img" in res:
            st.markdown('<div class="sec-hdr">🛰️ Input Satellite Scenes</div>', unsafe_allow_html=True)
            ic1, ic2 = st.columns(2)
            with ic1: st.image(res["before_img"], caption="Before (Time Period 1)", use_container_width=True)
            with ic2: st.image(res["after_img"],  caption="After (Time Period 2)",  use_container_width=True)
            st.markdown("<br>", unsafe_allow_html=True)

        # Land cover maps
        st.markdown('<div class="sec-hdr">🗺️ Land Cover Maps</div>', unsafe_allow_html=True)
        lc1, lc2 = st.columns(2)
        with lc1:
            fig = make_landcover_fig(mb, ps, "Land Cover — Before")
            st.pyplot(fig, use_container_width=True); plt.close(fig)
        with lc2:
            fig = make_landcover_fig(ma, ps, "Land Cover — After")
            st.pyplot(fig, use_container_width=True); plt.close(fig)

        # Deforestation map
        st.markdown('<div class="sec-hdr">🔴 Deforestation Detection Map</div>', unsafe_allow_html=True)
        fig = make_deforestation_fig(mb, ma, changes, ps)
        st.pyplot(fig, use_container_width=True); plt.close(fig)

        # Transition breakdown
        if changes:
            st.markdown('<div class="sec-hdr">🌲 Forest Loss Transitions</div>', unsafe_allow_html=True)
            tc = {}
            for ch in changes: tc[ch["to_class"]] = tc.get(ch["to_class"], 0) + 1
            total_ev = len(changes)
            cb, cp = st.columns(2)
            with cb:
                for cls, cnt in sorted(tc.items(), key=lambda x: -x[1]):
                    pct = cnt / total_ev * 100
                    st.markdown(f'<div class="trans-row"><span style="color:#8b949e;font-size:0.82rem;min-width:130px;">Forest → {cls}</span><div class="trans-bar-bg"><div class="trans-bar-fill" style="width:{pct:.1f}%;"></div></div><span style="color:#e6edf3;font-size:0.82rem;font-weight:600;min-width:72px;text-align:right;">{cnt} ({pct:.0f}%)</span></div>', unsafe_allow_html=True)
            with cp:
                fig = make_pie(tc); st.pyplot(fig, use_container_width=True); plt.close(fig)
        else:
            st.markdown('<div class="ok-box">✅ No deforestation events detected in this comparison.</div>', unsafe_allow_html=True)

        # Distribution table
        import pandas as pd
        st.markdown('<div class="sec-hdr">📋 Land Cover Distribution</div>', unsafe_allow_html=True)
        rows = []
        for cls in CLASS_NAMES:
            bc = int(np.sum(mb == cls)); ac = int(np.sum(ma == cls)); ch = ac - bc
            rows.append({"Class": cls, "Before (patches)": bc, "After (patches)": ac,
                         "Change": f"{ch:+d}" if ch != 0 else "—"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Download
        st.markdown('<div class="sec-hdr">📥 Download Report</div>', unsafe_allow_html=True)
        st.download_button(
            label="⬇️  Download Analysis Report (.txt)",
            data=make_report(changes, stats, mb, ma),
            file_name="deforestation_report.txt", mime="text/plain",
        )


# ───────────────────────────────────────────────────────────────
# TAB 3 — MODEL METRICS
# ───────────────────────────────────────────────────────────────
with tab_model_tab:
    st.markdown('<div class="sec-hdr">🧠 Architecture & Training</div>', unsafe_allow_html=True)
    m1, m2 = st.columns(2)
    with m1:
        st.markdown("""<div class="model-card">
          <div style="color:#e6edf3;font-weight:600;margin-bottom:10px;">Architecture</div>
          <div class="mc-row"><span class="mc-k">Base model</span><span class="mc-v">ResNet-50</span></div>
          <div class="mc-row"><span class="mc-k">Pre-training</span><span class="mc-v">ImageNet</span></div>
          <div class="mc-row"><span class="mc-k">Dataset</span><span class="mc-v">EuroSAT (27,000 images)</span></div>
          <div class="mc-row"><span class="mc-k">Output classes</span><span class="mc-v">10 land cover types</span></div>
          <div class="mc-row"><span class="mc-k">Input resolution</span><span class="mc-v">224 × 224 px</span></div>
          <div class="mc-row"><span class="mc-k">Trainable params</span><span class="mc-v">FC head only</span></div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown("""<div class="model-card">
          <div style="color:#e6edf3;font-weight:600;margin-bottom:10px;">Training Config</div>
          <div class="mc-row"><span class="mc-k">Epochs</span><span class="mc-v">10</span></div>
          <div class="mc-row"><span class="mc-k">Batch size</span><span class="mc-v">32</span></div>
          <div class="mc-row"><span class="mc-k">Optimizer</span><span class="mc-v">Adam (lr = 0.001)</span></div>
          <div class="mc-row"><span class="mc-k">Loss</span><span class="mc-v">CrossEntropyLoss</span></div>
          <div class="mc-row"><span class="mc-k">Split</span><span class="mc-v">80% / 10% / 10%</span></div>
          <div class="mc-row"><span class="mc-k">Random seed</span><span class="mc-v">42 (reproducible)</span></div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    cm_path   = os.path.join(ROOT, "outputs", "confusion_matrix.png")
    eval_path = os.path.join(ROOT, "outputs", "evaluation_report.txt")

    if os.path.exists(cm_path):
        st.markdown('<div class="sec-hdr">🔢 Confusion Matrix</div>', unsafe_allow_html=True)
        st.image(cm_path, use_container_width=True)
    else:
        st.markdown('<div class="info-box">ℹ️ Confusion matrix not found. Run <code>python src/evaluate.py</code> to generate it.</div>', unsafe_allow_html=True)

    if os.path.exists(eval_path):
        st.markdown('<div class="sec-hdr">📄 Evaluation Report</div>', unsafe_allow_html=True)
        with open(eval_path, "r") as f:
            st.code(f.read(), language=None)

    st.markdown('<div class="sec-hdr">📚 About EuroSAT</div>', unsafe_allow_html=True)
    st.markdown("""
[EuroSAT](https://github.com/phelber/EuroSAT) is a benchmark satellite image dataset based on **Sentinel-2** imagery —
**27,000 labeled images** (64×64 px) across 10 land use and land cover classes.

| Class | Description | Deforestation Target? |
|---|---|:---:|
| Forest | Dense tree cover | — |
| AnnualCrop | Seasonal agricultural fields | ✅ |
| HerbaceousVegetation | Grasslands, meadows | — |
| Residential | Urban residential areas | ✅ |
| Industrial | Factories, warehouses | ✅ |
| Pasture | Grazing land | ✅ |
| PermanentCrop | Orchards, vineyards | ✅ |
| Highway | Roads, motorways | — |
| River | Flowing water | — |
| SeaLake | Still water bodies | — |
""")