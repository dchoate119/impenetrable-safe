"""Render a ~90 s stop-motion video of how FingerNet works, using model.pt and the real data.

Run from anywhere:  python make_video.py
Writes the mp4 and a shot list (shots.txt) to --out-dir.
"""
import argparse
import shutil
import subprocess
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn as nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
IMG, TEST_FRAC, NUM_CLASSES = 64, 0.2, 6
FPS = 4                  # stop-motion rate: one new picture every 1/4 s
W, H = 1280, 720

PAPER, INK, MUTED = "#f3ede2", "#2b2b2b", "#8a8378"
RED, BLUE = "#c8553d", "#2f6690"
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.edgecolor": INK,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "axes.facecolor": PAPER})
CLASS_NAMES = ["0  fist / gap", "1 finger", "2 fingers", "3 fingers", "4 fingers", "5 fingers"]


def cmap(name):
    c = plt.get_cmap(name).copy()
    c.set_bad(PAPER)
    return c


MAGMA, DIVERGE, BLUES = cmap("magma"), cmap("RdBu_r"), cmap("Blues")


# --- Model + data (same as train.ipynb) --------------------------------------
def block(c_in, c_out):
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out),
                         nn.ReLU(), nn.MaxPool2d(2))


class FingerNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(block(1, 16), block(16, 32), block(32, 64), block(64, 128))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        return self.fc(self.dropout(self.pool(self.features(x)).flatten(1)))


def load_split():
    """Same split as train.ipynb. Returns test images/labels/paths and train counts."""
    Xte, yte, paths = [], [], []
    ntr = np.zeros(NUM_CLASSES, int)
    for session in sorted(p for p in DATA_DIR.iterdir() if p.is_dir()):
        for c in range(NUM_CLASSES):
            files = sorted((session / str(c)).glob("*.png"))
            cut = int(len(files) * (1 - TEST_FRAC))
            ntr[c] += cut
            for f in files[cut:]:
                img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
                Xte.append(cv2.resize(img, (IMG, IMG), interpolation=cv2.INTER_AREA))
                yte.append(c)
                paths.append(f)
    return np.array(Xte), np.array(yte), paths, ntr


def trace(model, img):
    """Run one 64x64 uint8 image through the net, keeping every intermediate."""
    h = torch.from_numpy(img.astype(np.float32) / 255.0)[None, None]
    acts = []
    with torch.no_grad():
        for blk in model.features:
            c = blk[0](h); b = blk[1](c); r = blk[2](b); p = blk[3](r)
            acts.append({k: v[0].numpy() for k, v in
                         dict(conv=c, bn=b, relu=r, pool=p).items()})
            h = p
        g = model.pool(h).flatten(1)
        s = model.fc(g)
        pr = torch.softmax(s, 1)
    return acts, g[0].numpy(), s[0].numpy(), pr[0].numpy()


# --- Drawing helpers -----------------------------------------------------------
class Reel:
    """Writes frames with a little camera shake + flicker so it reads as stop-motion."""

    def __init__(self, frame_dir):
        self.dir, self.n, self.marks = frame_dir, 0, []
        self.rng = np.random.default_rng(0)

    def mark(self, name):
        self.marks.append((self.n / FPS, name))

    def add(self, img, frames=1):
        for _ in range(frames):
            dx, dy = self.rng.integers(-2, 3, 2)
            M = cv2.getRotationMatrix2D((W / 2, H / 2), self.rng.uniform(-0.25, 0.25), 1.0)
            M[:, 2] += (dx, dy)
            f = cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REPLICATE)
            f = np.clip(f.astype(np.int16) + self.rng.integers(-4, 5), 0, 255).astype(np.uint8)
            cv2.imwrite(str(self.dir / f"{self.n:05d}.png"), f)
            self.n += 1


def new_fig(title=None, sub=None):
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=PAPER)
    if title:
        fig.text(0.04, 0.91, title, fontsize=30, weight="bold")
    if sub:
        fig.text(0.04, 0.855, sub, fontsize=16, color=MUTED)
    return fig


def grab(fig):
    fig.canvas.draw()
    a = np.asarray(fig.canvas.buffer_rgba())[..., :3]
    plt.close(fig)
    return cv2.cvtColor(a, cv2.COLOR_RGB2BGR)


def px(x, y, w, h):
    """Axes rect from pixel coords (x, y = bottom-left)."""
    return [x / W, y / H, w / W, h / H]


def show(fig, rect, arr, cm="gray", vmin=None, vmax=None, title=None, fs=14):
    ax = fig.add_axes(rect)
    ax.imshow(arr, cmap=cm, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(2)
    if title:
        ax.set_title(title, fontsize=fs)
    return ax


def arrow(fig, x0, y0, x1, y1):
    """Arrow in pixel coords."""
    fig.add_artist(FancyArrowPatch((x0 / W, y0 / H), (x1 / W, y1 / H), transform=fig.transFigure,
                                   arrowstyle="-|>", mutation_scale=28, lw=3, color=INK))


def text(fig, x, y, s, **kw):
    kw.setdefault("fontsize", 15)
    kw.setdefault("ha", "center")
    fig.text(x / W, y / H, s, **kw)


def mosaic(maps, cols, shown=None, signed=False, order=None, pad=1):
    """Tile (C, h, w) maps into one image, each map normalized on its own. NaN = blank."""
    C, h, w = maps.shape
    rows = -(-C // cols)
    out = np.full((rows * (h + pad) - pad, cols * (w + pad) - pad), np.nan)
    order = np.arange(C) if order is None else order
    for k in range(min(C, C if shown is None else shown)):
        m = maps[order[k]]
        mx = np.abs(m).max() if signed else m.max()
        r, c = divmod(k, cols)
        out[r * (h + pad):r * (h + pad) + h, c * (w + pad):c * (w + pad) + w] = m / mx if mx > 0 else 0
    return out


def by_strength(maps):
    return np.argsort(-maps.reshape(len(maps), -1).mean(1))


def prob_bars(fig, rect, probs, title, grow=1.0, ylim=(0, 1), fmt="{:.0%}"):
    ax = fig.add_axes(rect)
    win = int(np.argmax(probs))
    colors = [RED if i == win else BLUE for i in range(len(probs))]
    ax.bar(range(len(probs)), probs * grow, color=colors, width=0.7)
    ax.set_ylim(*ylim)
    ax.axhline(0, color=INK, lw=1)
    ax.set_xticks(range(len(probs)))
    ax.set_title(title, fontsize=15)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if grow >= 1.0 and fmt:
        v = probs[win]
        ax.text(win, v + (ylim[1] - ylim[0]) * 0.03 if v >= 0 else v - (ylim[1] - ylim[0]) * 0.08,
                fmt.format(v), ha="center", fontsize=14, color=RED, weight="bold")
    ax.set_xlabel("class", fontsize=13)
    return ax


# --- Scenes ----------------------------------------------------------------------
def scene_title(reel, crop128, x64):
    reel.mark("1. Title + input image")
    fig = new_fig()
    text(fig, W / 2, 400, "FingerNet", fontsize=60, weight="bold")
    text(fig, W / 2, 330, "a from-scratch CNN that counts fingers", fontsize=24, color=MUTED)
    reel.add(grab(fig), 16)
    title, sub = "What the network sees", "the fixed box on the webcam image is cropped and shrunk"
    for step in range(3):
        fig = new_fig(title, sub)
        show(fig, px(110, 140, 400, 400), crop128, vmin=0, vmax=255, title="camera box crop  128×128")
        if step >= 1:
            arrow(fig, 540, 340, 740, 340)
            text(fig, 640, 370, "shrink to 64×64", fontsize=15)
            text(fig, 640, 290, "pixels scaled to 0–1", fontsize=15)
        if step >= 2:
            show(fig, px(770, 140, 400, 400), x64, vmin=0, vmax=255, title="network input  1×64×64")
        reel.add(grab(fig), 8 if step < 2 else 16)


def scene_data(reel, Xte, examples, ntr, nte):
    reel.mark("2. The six classes (real data)")
    title = "The data"
    sub = f"{ntr.sum() + nte.sum():,} webcam crops · train on the first 80% of each recording, test on the last 20%"
    for k in range(1, NUM_CLASSES + 1):
        fig = new_fig(title, sub)
        for c in range(k):
            x = 50 + c * 200
            show(fig, px(x, 260, 180, 180), Xte[examples[c]], vmin=0, vmax=255,
                 title=CLASS_NAMES[c], fs=15)
            text(fig, x + 90, 220, f"train {ntr[c]}", fontsize=13, color=MUTED)
            text(fig, x + 90, 195, f"test {nte[c]}", fontsize=13, color=MUTED)
        reel.add(grab(fig), 5 if k < NUM_CLASSES else 19)


def scene_conv(reel, model, x64, acts):
    reel.mark("3. Convolution: a 3×3 filter slides over the image")
    W1 = model.features[0][0].weight.detach().numpy()[:, 0]   # (16, 3, 3)
    conv = acts[0]["conv"]                                      # (16, 64, 64)
    ch = int(np.argmax(conv.reshape(16, -1).std(1)))
    k, fmap = W1[ch], conv[ch]
    vmax = np.abs(fmap).max()
    title = "Layer 1: convolution"
    sub = "a 3×3 grid of learned weights slides over every pixel and makes one feature map"
    positions = [(int(r), int(c)) for r in np.linspace(3, 60, 6) for c in np.linspace(3, 60, 4)]
    for (r, c) in positions + [(63, 63)]:
        fig = new_fig(title, sub)
        ax = show(fig, px(60, 110, 380, 380), x64, vmin=0, vmax=255, title="input 64×64")
        ax.add_patch(Rectangle((c - 1.5, r - 1.5), 3, 3, fill=False, ec=RED, lw=3))
        kax = show(fig, px(540, 230, 200, 200), k, cm=DIVERGE, vmin=-np.abs(k).max(),
                   vmax=np.abs(k).max(), title="learned 3×3 weights")
        for i in range(3):
            for j in range(3):
                kax.text(j, i, f"{k[i, j]:+.2f}", ha="center", va="center", fontsize=12)
        text(fig, 640, 175, "Σ (pixel × weight) + bias", fontsize=15)
        text(fig, 640, 145, "= one output pixel", fontsize=15)
        arrow(fig, 455, 300, 525, 300)
        arrow(fig, 755, 300, 825, 300)
        part = fmap.copy()
        part.flat[r * 64 + c + 1:] = np.nan
        oax = show(fig, px(840, 110, 380, 380), part, cm=DIVERGE, vmin=-vmax, vmax=vmax,
                   title="feature map 64×64")
        oax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, ec=RED, lw=3))
        reel.add(grab(fig), 8 if (r, c) == (63, 63) else 1)
    fig = new_fig(title, "16 different filters → 16 feature maps, each picks out a different pattern")
    show(fig, px(80, 110, 380, 380), mosaic(W1, 4, signed=True, pad=1), cm=DIVERGE, vmin=-1, vmax=1,
         title="all 16 learned filters")
    arrow(fig, 500, 300, 760, 300)
    text(fig, 630, 330, "slide each over the image", fontsize=15)
    show(fig, px(800, 110, 380, 380), mosaic(conv, 4, signed=True), cm=DIVERGE, vmin=-1, vmax=1,
         title="16 feature maps")
    reel.add(grab(fig), 28)


def scene_block1(reel, acts):
    reel.mark("4. Block 1 step by step: conv → BatchNorm → ReLU → MaxPool")
    a = acts[0]
    chans = by_strength(a["relu"])[:3]
    cols = [("conv", "conv 3×3", "16 × 64×64", "weights × pixels"),
            ("bn", "BatchNorm", "16 × 64×64", "rescale: mean 0, spread 1"),
            ("relu", "ReLU", "16 × 64×64", "negatives → 0  (max(0, x))"),
            ("pool", "MaxPool 2×2", "16 × 32×32", "keep the max of each 2×2")]
    title = "Block 1, step by step"
    sub = "three of the 16 maps · red = positive, blue = negative"
    for n in range(1, 5):
        fig = new_fig(title, sub)
        for j in range(n):
            key, name, size, cap = cols[j]
            x = 150 + j * 280
            text(fig, x + 67, 575, name, fontsize=18, weight="bold")
            text(fig, x + 67, 550, size, fontsize=13, color=MUTED)
            text(fig, x + 67, 55, cap, fontsize=13)
            for i, ch in enumerate(chans):
                m = a[key][ch]
                ref = np.abs(a["bn"][ch]).max() if key in ("bn", "relu", "pool") else np.abs(m).max()
                show(fig, px(x, 395 - i * 145, 135, 135), m, cm=DIVERGE, vmin=-ref, vmax=ref)
            if j:
                arrow(fig, x - 130, 317, x - 15, 317)
        reel.add(grab(fig), 10 if n < 4 else 22)


def scene_deeper(reel, acts):
    reel.mark("5. Blocks 2–4: maps shrink, channels grow")
    title = "Going deeper"
    sub = "every block repeats conv → BatchNorm → ReLU → MaxPool · maps get smaller, patterns more abstract"
    specs = [(0, 4, "block 1\n16 × 32×32"), (1, 4, "block 2\n32 × 16×16"),
             (2, 8, "block 3\n64 × 8×8"), (3, 8, "block 4\n128 × 4×4")]
    for b in range(4):
        for step in range(1, 6):
            fig = new_fig(title, sub)
            for j in range(b + 1):
                blk, cols, label = specs[j]
                maps = acts[blk]["pool"]
                shown = None if j < b else int(len(maps) * step / 5)
                x = 70 + j * 300
                show(fig, px(x, 70, 240, 480), mosaic(maps, cols, shown, order=by_strength(maps)),
                     cm=MAGMA, vmin=0, vmax=1, title=label, fs=16)
                if j:
                    arrow(fig, x - 55, 310, x + 5, 310)
            reel.add(grab(fig), 4 if (b, step) != (3, 5) else 12)


def scene_head(reel, g, s, p):
    reel.mark("6. Average pool → linear → softmax")
    grid = g[np.argsort(-g)].reshape(16, 8) / g.max()
    smax = np.abs(s).max() * 1.25

    def draw(panel, frac):
        fig = new_fig("From maps to a decision",
                      "block 4 output (128 × 4×4) is squeezed into one prediction")
        ga = grid.copy()
        if panel == 0:
            ga.flat[int(128 * frac):] = np.nan
        show(fig, px(80, 110, 220, 440), ga, cm=MAGMA, vmin=0, vmax=1,
             title="average each map\n128 maps → 128 numbers", fs=15)
        text(fig, 190, 70, "(dropout 30%: training only)", fontsize=12, color=MUTED)
        if panel >= 1:
            arrow(fig, 330, 330, 440, 330)
            prob_bars(fig, px(470, 130, 330, 380), s, "linear layer: 128 → 6\nraw scores",
                      grow=frac if panel == 1 else 1.0, ylim=(-smax, smax), fmt="{:+.1f}")
        if panel >= 2:
            arrow(fig, 830, 330, 900, 330)
            prob_bars(fig, px(920, 130, 320, 380), p, "softmax\nprobabilities", grow=frac)
            if frac >= 1.0:
                text(fig, 1080, 40, f"prediction: {CLASS_NAMES[int(np.argmax(p))]}",
                     fontsize=20, color=RED, weight="bold")
        return grab(fig)

    for panel in range(3):
        for step in range(1, 4):
            reel.add(draw(panel, step / 4), 2)
        reel.add(draw(panel, 1.0), 12 if panel < 2 else 14)


def scene_all_classes(reel, model, Xte, examples):
    reel.mark("7. Every class through the same network")
    for c in range(NUM_CLASSES):
        img = Xte[examples[c]]
        acts, _, _, p = trace(model, img)
        maps = acts[3]["pool"]
        fig = new_fig("Every class, same network", "same weights, different activations, different answer")
        show(fig, px(90, 140, 340, 340), img, vmin=0, vmax=255, title=f"true: {CLASS_NAMES[c]}", fs=16)
        arrow(fig, 450, 310, 540, 310)
        show(fig, px(550, 100, 200, 420), mosaic(maps, 8, order=by_strength(maps)), cm=MAGMA,
             vmin=0, vmax=1, title="block 4 maps", fs=15)
        arrow(fig, 770, 310, 850, 310)
        prob_bars(fig, px(880, 140, 340, 340), p, "probabilities")
        text(fig, 1050, 45, f"predicted: {CLASS_NAMES[int(np.argmax(p))]}", fontsize=20,
             color=RED, weight="bold")
        reel.add(grab(fig), 7 if c < NUM_CLASSES - 1 else 9)


def plot_bounds(img):
    """Find the left/right spine columns and top/bottom spine rows of the two axes in loss.png."""
    dark = img.mean(2) < 100
    cols = np.where(dark.sum(0) > 0.6 * img.shape[0])[0]
    groups = np.split(cols, np.where(np.diff(cols) > 5)[0] + 1)
    xs = [int(g.mean()) for g in groups if len(g)]
    rows = np.where(dark.sum(1) > 0.35 * img.shape[1])[0]
    rgroups = np.split(rows, np.where(np.diff(rows) > 5)[0] + 1)
    ys = [int(g.mean()) for g in rgroups if len(g)]
    if len(xs) != 4 or len(ys) != 2:
        h, w = img.shape[:2]   # fallback: matches the 1320x480 loss.png
        xs = [int(w * .049), int(w * .487), int(w * .543), int(w * .986)]
        ys = [int(h * .09), int(h * .85)]
    return xs, ys


def scene_training(reel):
    reel.mark("8. Training curves")
    png = cv2.cvtColor(cv2.imread(str(HERE / "loss.png")), cv2.COLOR_BGR2RGB)
    xs, (top, bot) = plot_bounds(png)
    title = "Training"
    sub = "50 epochs · Adam · cross-entropy loss · random flips, rotations, brightness · best epoch kept"
    ih, iw = png.shape[:2]
    disp_w = 1200
    disp_h = disp_w * ih / iw
    n = 30
    for k in range(n + 14):
        f = min(k / (n - 1), 1.0)
        im = png.copy()
        for l, r in ((xs[0], xs[1]), (xs[2], xs[3])):
            cut = int(l + (r - l) * f) + 1
            if cut < r - 1:
                im[top + 2:bot - 1, cut:r - 1] = 255
        fig = new_fig(title, sub)
        ax = fig.add_axes(px(40, 40, disp_w, disp_h))
        ax.imshow(im); ax.axis("off")
        reel.add(grab(fig), 1)


def scene_results(reel, cm, acc, n_test, n_params):
    reel.mark("9. Test confusion matrix + accuracy")
    title = "Results on held-out test data"
    sub = "rows = true class, columns = predicted class · diagonal = correct"
    for rows in list(range(1, NUM_CLASSES + 1)) + ["final"]:
        fig = new_fig(title, sub)
        r = NUM_CLASSES if rows == "final" else rows
        shown = cm.astype(float).copy()
        shown[r:] = np.nan
        ax = show(fig, px(150, 60, 520, 520), shown, cm=BLUES, vmin=0, vmax=cm.max())
        ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
        ax.set_xlabel("predicted", fontsize=14); ax.set_ylabel("true", fontsize=14)
        for i in range(r):
            for j in range(NUM_CLASSES):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14,
                        color="white" if cm[i, j] > cm.max() / 2 else INK)
        if rows == "final":
            text(fig, 960, 400, f"{acc:.1%}", fontsize=72, weight="bold", color=RED)
            text(fig, 960, 340, f"test accuracy on {n_test} images", fontsize=18)
            text(fig, 960, 230, f"{n_params:,} parameters", fontsize=18, color=MUTED)
            text(fig, 960, 195, "exported to ONNX, runs on the UNO Q", fontsize=18, color=MUTED)
        reel.add(grab(fig), 4 if rows != "final" else 20)


# --- Main ------------------------------------------------------------------------------
def save_results(out_dir, cm, probs, preds, yte, ntr, nte, n_params):
    """Write results.txt and confusion_matrix.png for the report."""
    acc = (preds == yte).mean()
    conf = probs.max(1)
    right = preds == yte
    onnx = HERE.parent / "python" / "model.onnx"
    L = ["FingerNet test results", "=" * 22, "",
         f"test accuracy:  {acc:.1%}  ({right.sum()} / {len(yte)} images)",
         f"parameters:     {n_params:,}",
         f"ONNX size:      {onnx.stat().st_size / 1024:.0f} KB" if onnx.exists() else "ONNX size:      n/a",
         f"train images:   {ntr.sum()}   test images: {nte.sum()}",
         f"mean confidence when right: {conf[right].mean():.1%}   when wrong: "
         + (f"{conf[~right].mean():.1%}" if (~right).any() else "n/a"), "",
         "Per class            train  test  precision  recall    F1"]
    for c in range(NUM_CLASSES):
        tp = cm[c, c]
        prec = tp / cm[:, c].sum() if cm[:, c].sum() else 0.0
        rec = tp / cm[c].sum() if cm[c].sum() else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        L.append(f"  {CLASS_NAMES[c]:<17} {ntr[c]:5d} {nte[c]:5d}     {prec:6.1%}  {rec:6.1%}  {f1:5.2f}")
    L += ["", "Confusion matrix (rows = true, columns = predicted)",
          "       " + "".join(f"{c:6d}" for c in range(NUM_CLASSES))]
    L += [f"  {c:3d}  " + "".join(f"{v:6d}" for v in cm[c]) for c in range(NUM_CLASSES)]
    off = [(cm[i, j], i, j) for i in range(NUM_CLASSES) for j in range(NUM_CLASSES) if i != j and cm[i, j]]
    L += ["", "Most common mistakes"]
    L += [f"  true {i} → predicted {j}: {n}" for n, i, j in sorted(off, reverse=True)[:5]] or ["  none"]
    (out_dir / "results.txt").write_text("\n".join(L) + "\n")

    fig, ax = plt.subplots(figsize=(5, 4.5), dpi=150)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    im = ax.imshow(cm, cmap="Blues")
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set(xticks=range(NUM_CLASSES), yticks=range(NUM_CLASSES), xlabel="predicted", ylabel="true",
           title=f"Test confusion matrix (acc {acc:.1%})")
    fig.colorbar(im); fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png")
    plt.close(fig)
    print("\n".join(L))


def main(out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_dir = out_dir / "_frames"
    shutil.rmtree(frame_dir, ignore_errors=True)
    frame_dir.mkdir()

    model = FingerNet()
    model.load_state_dict(torch.load(HERE / "model.pt", map_location="cpu")["state_dict"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())

    Xte, yte, paths, ntr = load_split()
    nte = np.bincount(yte, minlength=NUM_CLASSES)
    with torch.no_grad():
        probs = torch.softmax(model(torch.from_numpy(Xte.astype(np.float32) / 255.0)[:, None]), 1).numpy()
    preds = probs.argmax(1)
    acc = (preds == yte).mean()
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), int)
    for t, q in zip(yte, preds):
        cm[t, q] += 1
    print(f"test accuracy {acc:.3f}\n{cm}")
    save_results(out_dir, cm, probs, preds, yte, ntr, nte, n_params)

    # Most confident correct test image of each class
    examples = []
    for c in range(NUM_CLASSES):
        idx = np.where(yte == c)[0]
        ok = idx[preds[idx] == c]
        pool = ok if len(ok) else idx
        examples.append(int(pool[np.argmax(probs[pool, c])]))

    demo = examples[3]
    x64 = Xte[demo]
    crop128 = cv2.imread(str(paths[demo]), cv2.IMREAD_GRAYSCALE)
    acts, g, s, p = trace(model, x64)

    reel = Reel(frame_dir)
    scene_title(reel, crop128, x64)
    scene_data(reel, Xte, examples, ntr, nte)
    scene_conv(reel, model, x64, acts)
    scene_block1(reel, acts)
    scene_deeper(reel, acts)
    scene_head(reel, g, s, p)
    scene_all_classes(reel, model, Xte, examples)
    scene_training(reel)
    scene_results(reel, cm, acc, len(yte), n_params)
    total = reel.n / FPS
    print(f"{reel.n} frames = {total:.1f} s")

    video = out_dir / "finger_net.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                    "-i", str(frame_dir / "%05d.png"), "-r", "30", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "18", str(video)], check=True)
    shutil.rmtree(frame_dir)

    marks = reel.marks + [(total, "end")]
    lines = [f"{int(t0 // 60)}:{t0 % 60:05.2f} – {int(t1 // 60)}:{t1 % 60:05.2f}  {name}"
             for (t0, name), (t1, _) in zip(marks, marks[1:])]
    (out_dir / "shots.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("saved", video)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path,
                    default=Path.home() / "Documents/ME-193/impenetrable-safe")
    main(ap.parse_args().out_dir)
