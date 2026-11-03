"""Shareable figures from a saved run (no GPU).

    python scripts/figures.py --run runs/v1 --out figures/
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "savefig.facecolor": SURFACE})


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def fig_behavior(run, out):
    s = json.load(open(os.path.join(run, "behavior", "summary.json")))["traits"]
    xs = [-1, -0.5, 0, 0.5, 1]
    fig, ax = plt.subplots(figsize=(8.4, 4.8), dpi=200)
    style(ax)
    for trait, amp, sup, color in (("openness", "o_amp", "o_sup", BLUE), ("neuroticism", "n_amp", "n_sup", ORANGE)):
        v = s[trait]
        for subset, ls, alpha, lab in (("own_set", "-", 0.16, "own prompts (n=40)"), ("general", "--", 0.0, "general prompts (n=104)")):
            states = [f"{amp}-1", f"{amp}-0.5", "base", f"{amp}+0.5", f"{amp}+1"]
            m = [v[st]["subsets"][subset]["score_mean"] for st in states]
            lo = [v[st]["subsets"][subset]["score_ci90"][0] for st in states]
            hi = [v[st]["subsets"][subset]["score_ci90"][1] for st in states]
            ax.plot(xs, m, ls, color=color, lw=2, marker="o" if ls == "-" else None, ms=7, mec=SURFACE, mew=1.5,
                    alpha=1 if ls == "-" else 0.75, zorder=3)
            if alpha:
                ax.fill_between(xs, lo, hi, color=color, alpha=alpha, lw=0, zorder=2)
            # independent suppressor and neutral control at the right
            for xpos, st, mk in ((1.55, f"{sup}+1", "D"), (1.95, "ctrl+1", "s")):
                e = v[st]["subsets"][subset]
                if ls == "-":
                    ax.errorbar([xpos + (0.04 if trait == "neuroticism" else -0.04)], [e["score_mean"]],
                                yerr=[[e["score_mean"] - e["score_ci90"][0]], [e["score_ci90"][1] - e["score_mean"]]],
                                fmt=mk, color=color, ms=7, mec=SURFACE, mew=1.5, elinewidth=1.5, capsize=0, zorder=3)
        ax.annotate(trait, (1, v[f"{amp}+1"]["subsets"]["own_set"]["score_mean"]), xytext=(8, 0),
                    textcoords="offset points", va="center", color=INK, fontsize=11, fontweight="bold")
    ax.axvline(1.3, color=GRID, lw=1)
    ax.set_xticks(xs + [1.55, 1.95])
    ax.set_xticklabels(["-1", "-0.5", "0\n(base)", "+0.5", "+1", "suppressor\nx1", "neutral\ncontrol x1"])
    ax.set_xlim(-1.2, 2.15)
    ax.set_ylim(1, 7)
    ax.set_xlabel("amplifier coefficient (effective LoRA update scaled; both factors untouched)")
    ax.set_ylabel("judged trait score (1-7)")
    ax.set_title("Qwen3-8B with Persona Cartography adapters: judged trait vs. adapter scale", loc="left", color=INK,
                 fontsize=12.5, pad=12)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color=INK2, lw=2, marker="o", ms=6, label="own trait prompts, n=40 (band: 90% CI)"),
                       Line2D([], [], color=INK2, lw=2, ls="--", label="general behavioral prompts, n=104"),
                       Line2D([], [], color=BLUE, lw=6, label="openness"),
                       Line2D([], [], color=ORANGE, lw=6, label="neuroticism")],
              loc="upper left", frameon=False, fontsize=9, labelcolor=INK2)
    fig.text(0.01, 0.01, "Blinded LLM judge (DeepSeek V3.2), greedy 160-token answers, one per prompt per state.",
             color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    p = os.path.join(out, "fig1_behavior_vs_scale.png")
    fig.savefig(p)
    print("wrote", p)


def fig_scatter(run, out, cand_path=None, scatter_path=None):
    z = np.load(scatter_path or os.path.join(run, "scatter_notemplate.npz"))
    index = json.load(open(cand_path or os.path.join(run, "neurons", "index.json")))
    tc = {(x["id"], x["trait"]): x for x in json.load(open(os.path.join(run, "template_check.json")))}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.0), dpi=200, sharex=True, sharey=True)
    groups = (("shared", "both poles", BLUE, "o"), ("amp_primary", "amplifier-primary", ORANGE, "^"),
              ("sup_primary", "suppressor-primary", AQUA, "v"))
    lo = 0.01
    for ax, trait in zip(axes, ("openness", "neuroticism")):
        x = np.clip(z[f"{trait}_amp"].astype(np.float32).ravel(), lo, None)
        y = np.clip(z[f"{trait}_sup"].astype(np.float32).ravel(), lo, None)
        hb = ax.hexbin(x, y, gridsize=70, xscale="log", yscale="log", bins="log", cmap="Greys", mincnt=1,
                       linewidths=0, extent=(np.log10(lo), np.log10(6), np.log10(lo), np.log10(6)))
        ax.plot([lo, 6], [lo, 6], color=MUTED, lw=1, ls=":", zorder=2)
        for g, lab, col, mk in groups:
            rs = [r for r in index if r.get("group", {}).get(trait) == g and (r["id"], trait) in tc]
            for hollow in (False, True):
                pts = [(tc[(r["id"], trait)]["zA_no_template"], tc[(r["id"], trait)]["zP_no_template"]) for r in rs
                       if (tc[(r["id"], trait)]["template_share"] > 0.5) == hollow]
                if pts:
                    px, py = zip(*pts)
                    ax.scatter(np.clip(px, lo, None), np.clip(py, lo, None), s=34, marker=mk,
                               facecolor="none" if hollow else col, edgecolor=col if hollow else SURFACE,
                               linewidth=1.2, zorder=4, label=None if hollow else lab)
        gen = [(tc[(r["id"], t2)]["zA_no_template"], tc[(r["id"], t2)]["zP_no_template"]) for r in index
               if "cross" in r.get("selected_for", []) for t2 in (trait,) if (r["id"], t2) in tc]
        if gen:
            px, py = zip(*gen)
            ax.scatter(px, py, s=90, facecolor="none", edgecolor=INK, linewidth=1.2, zorder=5, label="moved by every trait adapter")
        ax.set_title(trait, loc="left", color=INK, fontsize=12)
        ax.set_xlabel("change under amplifier x1 (RMS, in units of base RMS)")
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    axes[0].set_ylabel("change under suppressor x1 (same units)")
    axes[1].legend(loc="lower right", frameon=False, fontsize=8.5, labelcolor=INK2)
    fig.suptitle("Every MLP neuron in Qwen3-8B (36 x 12,288): how much each trait adapter changes it on identical text",
                 x=0.01, ha="left", color=INK, fontsize=12.5)
    fig.text(0.01, 0.01, "Gray density: all neurons (log count). Teacher-forced on a fixed 201k-token corpus, chat-template tokens "
             "excluded; change = RMS of (adapted - base) per token.\nHollow: selected neuron whose change was mostly on "
             "template tokens (it falls back into the bulk once they are excluded). Dotted: equal effect.", color=MUTED, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    p = os.path.join(out, "fig2_amp_vs_sup_all_neurons.png")
    fig.savefig(p)
    print("wrote", p)


def fig_components(run, out):
    d = json.load(open(os.path.join(run, "corr_vs_control_notemplate.json")))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), dpi=200, sharey=True)
    for ax, (trait, col) in zip(axes, (("openness", BLUE), ("neuroticism", ORANGE))):
        r, c = np.array(d[trait]["ratio"]), np.array(d[trait]["corr"])
        ax.hexbin(np.clip(r, 0, 1.6), c, gridsize=45, cmap="Greys", bins="log", mincnt=1, linewidths=0,
                  extent=(0, 1.6, -1, 1))
        edges = np.quantile(r, np.linspace(0, 1, 11))
        mid = [(np.median(r[(r >= a) & (r <= b)]), np.mean(c[(r >= a) & (r <= b)])) for a, b in zip(edges[:-1], edges[1:])]
        ax.plot(*zip(*mid), "-o", color=col, lw=2, ms=6, mec=SURFACE, mew=1.2, zorder=3, label="decile means")
        ax.axhline(0, color=MUTED, lw=1, ls=":")
        style(ax)
        ax.set_xlim(0, 1.6)
        ax.set_title(f"{trait} (top 1% most-changed neurons, n={len(r):,})", loc="left", color=INK, fontsize=11)
        ax.set_xlabel("neutral-control change / smaller trait-pole change")
    axes[0].set_ylabel("token-level correlation of\namplifier change and suppressor change")
    axes[0].legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK2)
    fig.suptitle("Where the neutral control also moves a neuron, both poles move it the same way;\n"
                 "where it doesn't, the amplifier and suppressor move it in opposite directions",
                 x=0.01, ha="left", color=INK, fontsize=11.5)
    fig.text(0.01, 0.01, "Each neuron: correlation over 200k teacher-forced tokens of (amp - base) with (sup - base). "
             "Chat-template tokens excluded. Control = recipe-matched neutral adapter, trained separately.", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 0.9))
    p = os.path.join(out, "fig3_shared_vs_trait_component.png")
    fig.savefig(p)
    print("wrote", p)


def fig_case(run, out):
    reads = json.load(open(os.path.join(run, "figdata", "L26_N8828_reads.json")))
    by = {"".join(r["tokens"]): r for r in reads}
    fig = plt.figure(figsize=(11, 5.4), dpi=200)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.18)
    axa, axb = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    # panel A: token strips, magnitude of the (negative) activation, one hue
    from matplotlib import colors as mcolors
    cmap = mcolors.LinearSegmentedColormap.from_list("mag", [SURFACE, "#e87ba4", "#8a1f4c"])
    rows = [("I miss you so much.", s) for s in ("base", "n_amp+1", "n_sup+1", "ctrl+1")] + \
           [("It was so difficult.", s) for s in ("base", "n_amp+1")] + [("It was very difficult.", s) for s in ("base", "n_amp+1")]
    labels = {"base": "base", "n_amp+1": "neuroticism amp x1", "n_sup+1": "neuroticism sup x1", "ctrl+1": "neutral control x1"}
    vmax = 22
    y = 0
    axa.set_xlim(0, 1)
    for k, (text, st) in enumerate(rows):
        r = by[text]
        if k in (4, 6):
            y += 0.6
        x = 0.34
        axa.text(0.0, -y, labels[st], va="center", fontsize=8.5, color=INK2)
        for tok, a in zip(r["tokens"], r["acts"][st]):
            w = 0.024 * max(len(tok), 1) + 0.02
            axa.add_patch(plt.Rectangle((x, -y - 0.38), w, 0.76, color=cmap(min(1, max(0, -a) / vmax)), lw=0))
            axa.text(x + w / 2, -y, tok.strip() or " ", ha="center", va="center", fontsize=9, color=INK)
            if -a > 9:
                axa.text(x + w / 2, -y - 0.62, f"{a:.1f}", ha="center", va="center", fontsize=6.5, color=INK2)
            x += w + 0.004
        y += 1.25
    axa.set_ylim(-y + 0.2, 0.8)
    axa.axis("off")
    axa.set_title("A. same text, different model states (darker = more negative)", loc="left", fontsize=10.5, color=INK)
    # panel B: activation at the intensifier vs amplifier coefficient
    xs = [-1, -0.5, 0, 0.5, 1]
    sts = ["n_amp-1", "n_amp-0.5", "base", "n_amp+0.5", "n_amp+1"]
    for (text, tok_i), col, lab in ((("I miss you so much.", 4), BLUE, '"so much"'), (("It was so difficult.", 3), ORANGE, '"so difficult"'),
                                    (("It was very difficult.", 3), AQUA, '"very difficult"')):
        r = by[text]
        v = [r["acts"][s_][tok_i] for s_ in sts]
        axb.plot(xs, v, "-o", color=col, lw=2, ms=6, mec=SURFACE, mew=1.2, label=lab)
        axb.scatter([1.5], [r["acts"]["n_sup+1"][tok_i]], marker="D", color=col, s=36, edgecolor=SURFACE, zorder=3)
        axb.scatter([1.9], [r["acts"]["ctrl+1"][tok_i]], marker="s", color=col, s=36, edgecolor=SURFACE, zorder=3)
    style(axb)
    axb.axvline(1.25, color=GRID, lw=1)
    axb.set_xticks(xs + [1.5, 1.9])
    axb.set_xticklabels(["-1", "-0.5", "0", "+0.5", "+1", "sup\nx1", "ctrl\nx1"])
    axb.set_xlabel("neuroticism amplifier coefficient")
    axb.set_ylabel("activation at the intensified word")
    axb.legend(frameon=False, fontsize=8.5, loc="lower left", labelcolor=INK2)
    axb.set_title("B. dose-response at the target token", loc="left", fontsize=10.5, color=INK)
    fig.suptitle("Neuron L26/N8828: a native response to intensified words after \"so\", scaled up by the neuroticism "
                 "amplifier and down by the suppressor", x=0.01, ha="left", fontsize=11.5, color=INK)
    fig.text(0.01, -0.03, "Live measurements on Qwen3-8B, plain text. Selectivity (so + word vs very + word) is kept; "
             "the adapters change its size. See runs/v1/profiles/L26_N8828.md.", color=MUTED, fontsize=8)
    p = os.path.join(out, "fig4_case_L26_N8828.png")
    fig.savefig(p, bbox_inches="tight")
    print("wrote", p)



def fig_matrix(run, out):
    """Four adapters side by side: whether each pair pushes neurons the same or opposite ways, and how much the
    neurons each adapter changes most overlap."""
    idx = {r["id"]: r for r in json.load(open(os.path.join(run, "neurons", "index.json")))}
    ct = {r["id"]: r for r in json.load(open(os.path.join(run, "cross_trait.json")))}
    A = ["o_amp+1", "o_sup+1", "n_amp+1", "n_sup+1"]
    names = ["openness\namplifier", "openness\nsuppressor", "neuroticism\namplifier", "neuroticism\nsuppressor"]
    T = {"o": "openness", "n": "neuroticism"}

    def corrs(a, b):
        # within a trait: that trait's neurons; across traits: neurons selected for either trait
        want = {T[a[0]], T[b[0]]}
        ids = [i for i, r in idx.items() if r["kind"] != "random_control" and want & set(r.get("population_traits") or [])]
        if a[0] == b[0]:
            v = [(idx[i].get("corr_amp_sup") or {}).get(T[a[0]]) for i in ids]
        else:
            k = f"corr_{a}|{b}|all" if a[0] == "o" else f"corr_{b}|{a}|all"
            v = [ct.get(i, {}).get(k) for i in ids]
        return np.array([x for x in v if x is not None], float)

    R = np.full((4, 4), np.nan)
    opp = np.full((4, 4), np.nan)
    nn = {}
    for x in range(4):
        for y in range(4):
            if x != y:
                v = corrs(A[x], A[y])
                R[x, y], opp[x, y], nn[(x, y)] = np.median(v), (v < 0).mean(), len(v)

    d = np.load(os.path.join(run, "scatter.npz"))
    Z = [d[k].astype(float).ravel() for k in ("openness_amp", "openness_sup", "neuroticism_amp", "neuroticism_sup",
                                               "control")]
    k = len(Z[0]) // 100
    top = [set(np.argsort(-z)[:k].tolist()) for z in Z]
    O = np.array([[len(top[i] & top[j]) / k for j in range(5)] for i in range(5)])

    from matplotlib.colors import LinearSegmentedColormap
    RED, BLUE2, MID = "#e34948", "#2a78d6", "#f0efec"
    div = LinearSegmentedColormap.from_list("div", [RED, MID, BLUE2])
    seq = LinearSegmentedColormap.from_list("seq", [MID, "#4b3cb0"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw={"width_ratios": [3, 4], "wspace": 0.35})

    low = lambda n: np.tril(np.ones((n, n), bool), -1)  # symmetric: lower triangle only
    ax1.imshow(np.where(low(4), R, np.nan), cmap=div, vmin=-0.6, vmax=0.6)
    for x in range(4):
        for y in range(x):
            ax1.text(y, x - 0.12, f"{opp[x, y] * 100:.0f}%", ha="center", va="center", fontsize=14, color=INK,
                     fontweight="bold")
            ax1.text(y, x + 0.2, f"opposite\nmedian r {R[x, y]:+.2f}", ha="center", va="center", fontsize=8,
                     color=INK2, linespacing=1.1)
    ax1.set_title("A. Same way or opposite ways?", loc="left", fontsize=11.5, color=INK)

    ax2.imshow(np.where(low(5), O, np.nan), cmap=seq, vmin=0, vmax=0.6)
    for i in range(5):
        for j in range(i):
            ax2.text(j, i, f"{O[i, j] * 100:.0f}%", ha="center", va="center", fontsize=12, color=INK)
    ax2.axhline(3.5, color=SURFACE, lw=5)
    ax2.set_title("B. Do they change the same neurons?", loc="left", fontsize=11.5, color=INK)

    for ax, labs in ((ax1, names), (ax2, names + ["neutral\ncontrol"])):
        ax.set_xticks(range(len(labs) - 1))
        ax.set_yticks(range(1, len(labs)))
        ax.set_xticklabels(labs[:-1], fontsize=9.5)
        ax.set_yticklabels(labs[1:], fontsize=9.5)
        ax.set_xlim(-0.5, len(labs) - 1.5)
        ax.set_ylim(len(labs) - 0.5, 0.5)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
    for ax in (ax1, ax2):
        for t in ax.get_xticklabels() + ax.get_yticklabels():
            if t.get_text().startswith("neutral"):
                t.set_color(MUTED)

    fig.suptitle("How the four personality adapters relate to each other at the neuron level", x=0.01, y=1.0,
                 ha="left", fontsize=13, color=INK)
    n_o, n_n, n_x = nn[(0, 1)], nn[(2, 3)], nn[(0, 2)]
    fig.text(0.01, -0.06,
             f"A: for each neuron, the correlation between the two adapters' changes to its activation over the "
             f"201k-token corpus; cells give the share of neurons where it is negative (the two push it opposite ways). "
             f"Within a trait: that trait's neurons\n(openness n={n_o:,}, neuroticism n={n_n:,}); across traits: "
             f"neurons selected for either trait (n={n_x:,}). Red = mostly opposite, blue = mostly same way. "
             f"B: overlap between the 1% of neurons each adapter changes most ({k:,} of {len(Z[0]):,}); chance is 1%.\n"
             "The neutral control, trained the same way with no personality goal, is shown for reference: much of the "
             "overlap is change that fine-tuning makes regardless of trait.", color=MUTED, fontsize=8, ha="left")
    pth = os.path.join(out, "fig5_adapter_matrix.png")
    fig.savefig(pth, bbox_inches="tight", dpi=150)
    print("wrote", pth)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/v1")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--only", default=None)
    ap.add_argument("--candidates", default=None)
    ap.add_argument("--scatter", default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    figs = {"matrix": fig_matrix, "behavior": fig_behavior, "components": fig_components, "case": fig_case,
            "scatter": lambda r, o: fig_scatter(r, o, args.candidates, args.scatter)}
    for name, f in figs.items():
        if args.only in (None, name):
            f(args.run, args.out)


if __name__ == "__main__":
    main()
