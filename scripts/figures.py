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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/v1")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    figs = {"behavior": fig_behavior}
    for name, f in figs.items():
        if args.only in (None, name):
            f(args.run, args.out)


if __name__ == "__main__":
    main()
