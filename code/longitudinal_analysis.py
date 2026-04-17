"""
Longitudinal Progression Analysis
===================================
Analyses:
  1. Progression Rate by Cluster
     - KL grade change from V00 → V03 → V05 → V06 → V08 → V10
     - Compare progression rate between Lateral JSN vs Medial JSN clusters
     - Kaplan-Meier style progression curves per phenotype

  2. Cluster Stability
     - Do subjects stay in the same compartment phenotype across visits?
     - Stability = same dominant compartment (lateral/medial/none) at follow-up

Outputs:
  /longitudinal/
    progression_rate_by_cluster.png
    kl_trajectory_by_cluster.png
    pain_trajectory_by_cluster.png
    cluster_stability.png
    stability_sankey.png
    longitudinal_summary.csv

Usage:
    python longitudinal_analysis.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR    = Path("/scratch1/e20-fyp-vlm-knee-osteo/FYP/Dataset")
CLUSTER_CSV = Path("/scratch1/e20-fyp-vlm-knee-osteo/e20378-4yp-VLM-Guided-Phenotyping-of-KOA/clustering/cluster_assignments_finetuned.csv")
OUTPUT_DIR  = Path("/scratch1/e20-fyp-vlm-knee-osteo/e20378-4yp-VLM-Guided-Phenotyping-of-KOA/longitudinal")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VISITS      = ["V00", "V03", "V05", "V06", "V08", "V10"]
VISIT_MONTHS = {"V00": 0, "V03": 36, "V05": 60, "V06": 72, "V08": 96, "V10": 120}

# Phenotype colors
PHENOTYPE_COLORS = {
    "Lateral JSN": "#e74c3c",   # red
    "Medial JSN":  "#2980b9",   # blue
    "No JSN":      "#27ae60",   # green
    "Pain-Dominant": "#f39c12", # orange (KL0 special)
    "Healthy":     "#95a5a6",   # gray
}


# ── Load and prepare data ──────────────────────────────────────────────────────
def load_all_visits():
    """Load all visit CSVs and standardize columns."""
    dfs = {}
    for v in VISITS:
        f = DATA_DIR / f"master_final_{v}.csv"
        if not f.exists():
            print(f"  Missing: {f.name}")
            continue
        df = pd.read_csv(f)
        df["SRC_SUBJECT_ID"] = df["SRC_SUBJECT_ID"].astype(str)
        df["VISIT"]          = v
        df["MONTHS"]         = VISIT_MONTHS[v]
        # Standardize column names
        df = df.rename(columns={
            "KL_GRADE":        "kl_grade",
            "KNEE_SIDE":       "knee_side",
            "JSN_LATERAL":     "jsn_lat",
            "JSN_MEDIAL":      "jsn_med",
            "MATCHED_WOMAC_PAIN": "pain",
            "BMI":             "bmi",
            "AGEYEARS":        "age",
        })
        dfs[v] = df
        print(f"  {v}: {len(df)} rows, {df['SRC_SUBJECT_ID'].nunique()} subjects")
    return dfs


def assign_phenotype(row):
    """Assign phenotype label based on cluster assignment."""
    kl  = row["kl_grade_V00"]
    cl  = row["cluster"]
    jl  = row.get("jsn_lat_V00", 0)
    jm  = row.get("jsn_med_V00", 0)
    pain = row.get("pain_V00", 0)

    if kl == 0:
        return "Pain-Dominant" if pain >= 2 else "Healthy"
    elif jl > jm:
        return "Lateral JSN"
    elif jm > jl:
        return "Medial JSN"
    else:
        return "No JSN"


def build_longitudinal_df(dfs, cluster_df):
    """
    Build wide-format longitudinal dataframe.
    One row per subject-knee, columns for each visit's KL grade, pain, JSN.
    """
    # Start with V00 cluster assignments
    base = cluster_df[cluster_df["kl_grade"].notna()].copy()
    base["SRC_SUBJECT_ID"] = base["subject_id"].astype(str)
    base["knee_side"]      = base["knee_side"]

    # Merge V00 metadata
    v00 = dfs["V00"][["SRC_SUBJECT_ID", "knee_side", "kl_grade",
                       "jsn_lat", "jsn_med", "pain", "bmi", "age"]].copy()
    v00.columns = ["SRC_SUBJECT_ID", "knee_side",
                   "kl_grade_V00", "jsn_lat_V00", "jsn_med_V00",
                   "pain_V00", "bmi_V00", "age_V00"]

    merged = base.merge(v00, on=["SRC_SUBJECT_ID", "knee_side"], how="left")

    # Merge follow-up visits
    for v in VISITS[1:]:
        if v not in dfs:
            continue
        vdf = dfs[v][["SRC_SUBJECT_ID", "knee_side",
                       "kl_grade", "jsn_lat", "jsn_med", "pain"]].copy()
        vdf.columns = ["SRC_SUBJECT_ID", "knee_side",
                       f"kl_grade_{v}", f"jsn_lat_{v}",
                       f"jsn_med_{v}", f"pain_{v}"]
        merged = merged.merge(vdf, on=["SRC_SUBJECT_ID", "knee_side"], how="left")

    # Assign phenotype
    merged["phenotype"] = merged.apply(assign_phenotype, axis=1)

    print(f"\nLongitudinal cohort: {len(merged)} knee records")
    print(f"Phenotype distribution:")
    print(merged["phenotype"].value_counts().to_string())

    return merged


# ── Analysis 1: Progression Rate ──────────────────────────────────────────────
def plot_kl_trajectory(long_df, output_dir):
    """
    Plot mean KL grade over time for each phenotype.
    Shows which phenotypes progress faster.
    """
    phenotypes = ["Lateral JSN", "Medial JSN", "No JSN",
                  "Pain-Dominant", "Healthy"]
    visit_cols  = [f"kl_grade_{v}" for v in VISITS if f"kl_grade_{v}" in long_df.columns]
    months      = [VISIT_MONTHS[v] for v in VISITS if f"kl_grade_{v}" in long_df.columns]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ── Plot 1: Mean KL grade trajectory ──────────────────────────────────
    for phenotype in phenotypes:
        sub = long_df[long_df["phenotype"] == phenotype]
        if len(sub) < 10:
            continue

        means = []
        cis   = []
        valid_months = []

        for col, month in zip(visit_cols, months):
            vals = sub[col].dropna()
            if len(vals) < 10:
                continue
            means.append(vals.mean())
            ci = 1.96 * vals.std() / np.sqrt(len(vals))
            cis.append(ci)
            valid_months.append(month)

        if len(means) < 2:
            continue

        color = PHENOTYPE_COLORS.get(phenotype, "gray")
        axes[0].plot(valid_months, means, "o-", color=color,
                    label=f"{phenotype} (n={len(sub)})", linewidth=2, markersize=6)
        axes[0].fill_between(valid_months,
                             [m-c for m,c in zip(means,cis)],
                             [m+c for m,c in zip(means,cis)],
                             alpha=0.15, color=color)

    axes[0].set_xlabel("Months from Baseline")
    axes[0].set_ylabel("Mean KL Grade")
    axes[0].set_title("KL Grade Progression by Phenotype")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(months)
    axes[0].set_xticklabels([f"V{m}" if m > 0 else "V00\n(baseline)"
                              for m in months], fontsize=8)

    # ── Plot 2: KL grade change from baseline ─────────────────────────────
    for phenotype in phenotypes:
        sub = long_df[long_df["phenotype"] == phenotype]
        if len(sub) < 10:
            continue

        changes       = []
        valid_months  = []

        baseline_col = "kl_grade_V00"
        for col, month in zip(visit_cols[1:], months[1:]):
            paired = sub[[baseline_col, col]].dropna()
            if len(paired) < 10:
                continue
            change = (paired[col] - paired[baseline_col]).mean()
            changes.append(change)
            valid_months.append(month)

        if len(changes) < 2:
            continue

        color = PHENOTYPE_COLORS.get(phenotype, "gray")
        axes[1].plot(valid_months, changes, "o-", color=color,
                    label=f"{phenotype}", linewidth=2, markersize=6)

    axes[1].axhline(y=0, color="black", linestyle="--", alpha=0.5)
    axes[1].set_xlabel("Months from Baseline")
    axes[1].set_ylabel("Mean KL Grade Change from Baseline")
    axes[1].set_title("KL Grade Change from Baseline by Phenotype")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(months[1:])

    plt.tight_layout()
    out = output_dir / "kl_trajectory_by_cluster.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"KL trajectory plot saved → {out.name}")


def plot_progression_rate(long_df, output_dir):
    """
    Bar chart: % of subjects who progressed by ≥1 and ≥2 KL grades
    by V08 (8 years), per phenotype.
    """
    if "kl_grade_V08" not in long_df.columns:
        print("V08 not available, skipping progression rate")
        return

    phenotypes = ["Lateral JSN", "Medial JSN", "No JSN",
                  "Pain-Dominant", "Healthy"]

    results = []
    for phenotype in phenotypes:
        sub = long_df[long_df["phenotype"] == phenotype]
        paired = sub[["kl_grade_V00", "kl_grade_V08"]].dropna()

        if len(paired) < 10:
            continue

        delta = paired["kl_grade_V08"] - paired["kl_grade_V00"]
        n     = len(paired)
        prog1 = (delta >= 1).sum() / n * 100
        prog2 = (delta >= 2).sum() / n * 100
        mean_delta = delta.mean()

        results.append({
            "phenotype":  phenotype,
            "n":          n,
            "prog_1grade": prog1,
            "prog_2grade": prog2,
            "mean_delta":  mean_delta,
        })

        print(f"\n{phenotype} (n={n}):")
        print(f"  Progressed ≥1 KL grade by V08: {prog1:.1f}%")
        print(f"  Progressed ≥2 KL grade by V08: {prog2:.1f}%")
        print(f"  Mean KL change: {mean_delta:.2f}")

    if not results:
        return

    df_res = pd.DataFrame(results)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    colors = [PHENOTYPE_COLORS.get(p, "gray") for p in df_res["phenotype"]]

    # ≥1 grade progression
    bars = axes[0].bar(df_res["phenotype"], df_res["prog_1grade"],
                       color=colors, edgecolor="white", linewidth=0.5)
    axes[0].set_title("% Progressed ≥1 KL Grade by V08 (8 years)")
    axes[0].set_ylabel("Percentage (%)")
    axes[0].set_ylim(0, 100)
    for bar, val in zip(bars, df_res["prog_1grade"]):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].grid(axis="y", alpha=0.3)

    # ≥2 grade progression
    bars = axes[1].bar(df_res["phenotype"], df_res["prog_2grade"],
                       color=colors, edgecolor="white", linewidth=0.5)
    axes[1].set_title("% Progressed ≥2 KL Grades by V08 (8 years)")
    axes[1].set_ylabel("Percentage (%)")
    axes[1].set_ylim(0, 100)
    for bar, val in zip(bars, df_res["prog_2grade"]):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].grid(axis="y", alpha=0.3)

    # Mean KL change
    bars = axes[2].bar(df_res["phenotype"], df_res["mean_delta"],
                       color=colors, edgecolor="white", linewidth=0.5)
    axes[2].set_title("Mean KL Grade Change by V08")
    axes[2].set_ylabel("Mean ΔKL Grade")
    axes[2].axhline(y=0, color="black", linewidth=0.5)
    for bar, val in zip(bars, df_res["mean_delta"]):
        axes[2].text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.02 if val >= 0 else bar.get_height() - 0.08,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    axes[2].tick_params(axis="x", rotation=30)
    axes[2].grid(axis="y", alpha=0.3)

    plt.suptitle("Progression Rate by Baseline Phenotype Cluster (V00 → V08)",
                 fontsize=13)
    plt.tight_layout()
    out = output_dir / "progression_rate_by_cluster.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nProgression rate plot saved → {out.name}")

    # Statistical test — Medial vs Lateral JSN progression
    lat = long_df[long_df["phenotype"] == "Lateral JSN"][
        ["kl_grade_V00", "kl_grade_V08"]].dropna()
    med = long_df[long_df["phenotype"] == "Medial JSN"][
        ["kl_grade_V00", "kl_grade_V08"]].dropna()

    if len(lat) > 10 and len(med) > 10:
        lat_delta = lat["kl_grade_V08"] - lat["kl_grade_V00"]
        med_delta = med["kl_grade_V08"] - med["kl_grade_V00"]
        t_stat, p_val = stats.mannwhitneyu(lat_delta, med_delta,
                                           alternative="two-sided")
        print(f"\nMann-Whitney U test — Lateral vs Medial JSN progression:")
        print(f"  Lateral mean ΔKL = {lat_delta.mean():.3f}")
        print(f"  Medial  mean ΔKL = {med_delta.mean():.3f}")
        print(f"  U={t_stat:.1f}  p={p_val:.4f}")
        if p_val < 0.05:
            print(f"  *** Statistically significant difference (p<0.05) ***")
        else:
            print(f"  Not statistically significant (p={p_val:.3f})")

    return df_res


# ── Analysis 2: Cluster Stability ─────────────────────────────────────────────
def get_compartment(row, visit):
    """Determine dominant compartment at a given visit."""
    jl_col = f"jsn_lat_{visit}"
    jm_col = f"jsn_med_{visit}"

    if jl_col not in row.index or jm_col not in row.index:
        return "Unknown"

    jl = row[jl_col]
    jm = row[jm_col]

    if pd.isna(jl) or pd.isna(jm):
        return "Unknown"

    if jl > jm:
        return "Lateral"
    elif jm > jl:
        return "Medial"
    else:
        return "None"


def plot_cluster_stability(long_df, output_dir):
    """
    Show what % of subjects remain in same compartment phenotype
    across follow-up visits.
    """
    # Only use subjects with lateral or medial JSN at baseline
    jsn_subjects = long_df[
        long_df["phenotype"].isin(["Lateral JSN", "Medial JSN"])
    ].copy()

    print(f"\nStability analysis: {len(jsn_subjects)} subjects with JSN at baseline")

    # Get compartment at each visit
    jsn_subjects["compartment_V00"] = jsn_subjects.apply(
        lambda r: get_compartment(r, "V00"), axis=1
    )

    stability_data = []

    for v in VISITS[1:]:
        jl_col = f"jsn_lat_{v}"
        jm_col = f"jsn_med_{v}"

        if jl_col not in jsn_subjects.columns:
            continue

        paired = jsn_subjects[
            ["compartment_V00", jl_col, jm_col]
        ].dropna()

        if len(paired) < 10:
            continue

        paired["compartment_followup"] = paired.apply(
            lambda r: "Lateral" if r[jl_col] > r[jm_col]
                      else ("Medial" if r[jm_col] > r[jl_col] else "None"),
            axis=1
        )

        # Stability = same compartment as baseline
        same = (paired["compartment_V00"] == paired["compartment_followup"]).mean() * 100

        # Per-phenotype stability
        for compartment in ["Lateral", "Medial"]:
            sub = paired[paired["compartment_V00"] == compartment]
            if len(sub) < 5:
                continue
            sub_same = (sub["compartment_V00"] == sub["compartment_followup"]).mean() * 100
            stability_data.append({
                "visit":       v,
                "months":      VISIT_MONTHS[v],
                "compartment": compartment,
                "stability":   sub_same,
                "n":           len(sub),
            })

        print(f"  {v}: overall stability={same:.1f}%  (n={len(paired)})")

    if not stability_data:
        print("Not enough data for stability analysis")
        return

    df_stab = pd.DataFrame(stability_data)

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Stability over time per compartment
    for compartment, color in [("Lateral", "#e74c3c"), ("Medial", "#2980b9")]:
        sub = df_stab[df_stab["compartment"] == compartment]
        if sub.empty:
            continue
        axes[0].plot(sub["months"], sub["stability"], "o-",
                    color=color, linewidth=2, markersize=8,
                    label=f"{compartment} JSN")
        for _, row in sub.iterrows():
            axes[0].annotate(f"n={int(row['n'])}",
                           (row["months"], row["stability"]),
                           textcoords="offset points", xytext=(0, 8),
                           fontsize=7, ha="center")

    axes[0].axhline(y=50, color="gray", linestyle="--", alpha=0.5,
                   label="50% (random chance)")
    axes[0].set_xlabel("Months from Baseline")
    axes[0].set_ylabel("% Subjects with Same Compartment Phenotype")
    axes[0].set_title("Compartment Phenotype Stability Over Time")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 105)
    axes[0].set_xticks(df_stab["months"].unique())

    # Grouped bar chart at V08
    v08_data = df_stab[df_stab["visit"] == "V08"]
    if not v08_data.empty:
        bars = axes[1].bar(
            v08_data["compartment"],
            v08_data["stability"],
            color=["#e74c3c", "#2980b9"],
            width=0.4, edgecolor="white"
        )
        for bar, (_, row) in zip(bars, v08_data.iterrows()):
            axes[1].text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 1,
                        f"{row['stability']:.1f}%\n(n={int(row['n'])})",
                        ha="center", va="bottom", fontsize=10)
        axes[1].set_ylim(0, 110)
        axes[1].set_ylabel("% Stable Compartment Phenotype")
        axes[1].set_title("Phenotype Stability at 8-Year Follow-up (V08)")
        axes[1].axhline(y=50, color="gray", linestyle="--", alpha=0.5)
        axes[1].grid(axis="y", alpha=0.3)

    plt.suptitle("Cluster Stability — Compartment Phenotype Persistence Over 10 Years",
                 fontsize=12)
    plt.tight_layout()
    out = output_dir / "cluster_stability.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Cluster stability plot saved → {out.name}")

    return df_stab


# ── Pain trajectory ────────────────────────────────────────────────────────────
def plot_pain_trajectory(long_df, output_dir):
    """Mean WOMAC pain score over time per phenotype."""
    phenotypes  = ["Lateral JSN", "Medial JSN", "No JSN",
                   "Pain-Dominant", "Healthy"]
    pain_cols   = [f"pain_{v}" for v in VISITS if f"pain_{v}" in long_df.columns]
    months      = [VISIT_MONTHS[v] for v in VISITS if f"pain_{v}" in long_df.columns]

    fig, ax = plt.subplots(figsize=(10, 6))

    for phenotype in phenotypes:
        sub = long_df[long_df["phenotype"] == phenotype]
        if len(sub) < 10:
            continue

        means, valid_months = [], []
        for col, month in zip(pain_cols, months):
            vals = sub[col].dropna()
            if len(vals) < 10:
                continue
            means.append(vals.mean())
            valid_months.append(month)

        if len(means) < 2:
            continue

        color = PHENOTYPE_COLORS.get(phenotype, "gray")
        ax.plot(valid_months, means, "o-", color=color,
               label=f"{phenotype} (n={len(sub)})", linewidth=2, markersize=6)

    ax.set_xlabel("Months from Baseline")
    ax.set_ylabel("Mean WOMAC Pain Score")
    ax.set_title("Pain Trajectory by Phenotype Over 10 Years")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(months)

    plt.tight_layout()
    out = output_dir / "pain_trajectory_by_cluster.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Pain trajectory plot saved → {out.name}")


# ── Summary CSV ────────────────────────────────────────────────────────────────
def save_summary(long_df, output_dir):
    """Save per-subject longitudinal summary to CSV."""
    kl_cols   = [c for c in long_df.columns if c.startswith("kl_grade_")]
    pain_cols = [c for c in long_df.columns if c.startswith("pain_")]
    save_cols = ["SRC_SUBJECT_ID", "knee_side", "phenotype",
                 "cluster", "kl_grade"] + kl_cols + pain_cols

    out = output_dir / "longitudinal_summary.csv"
    long_df[[c for c in save_cols if c in long_df.columns]].to_csv(out, index=False)
    print(f"Summary CSV saved → {out.name}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Longitudinal Progression Analysis")
    print("=" * 60)

    # Load data
    print("\nLoading visit CSVs...")
    dfs = load_all_visits()

    print("\nLoading cluster assignments...")
    cluster_df = pd.read_csv(CLUSTER_CSV)
    print(f"  {len(cluster_df)} cluster assignments loaded")

    # Build longitudinal dataframe
    print("\nBuilding longitudinal cohort...")
    long_df = build_longitudinal_df(dfs, cluster_df)

    # ── Analysis 1: KL Trajectory ──────────────────────────────────────────
    print("\n" + "="*60)
    print("ANALYSIS 1: KL GRADE TRAJECTORY")
    print("="*60)
    plot_kl_trajectory(long_df, OUTPUT_DIR)

    # ── Analysis 2: Progression Rate ──────────────────────────────────────
    print("\n" + "="*60)
    print("ANALYSIS 2: PROGRESSION RATE BY CLUSTER")
    print("="*60)
    df_prog = plot_progression_rate(long_df, OUTPUT_DIR)

    # ── Analysis 3: Cluster Stability ─────────────────────────────────────
    print("\n" + "="*60)
    print("ANALYSIS 3: CLUSTER STABILITY")
    print("="*60)
    df_stab = plot_cluster_stability(long_df, OUTPUT_DIR)

    # ── Analysis 4: Pain Trajectory ───────────────────────────────────────
    print("\n" + "="*60)
    print("ANALYSIS 4: PAIN TRAJECTORY")
    print("="*60)
    plot_pain_trajectory(long_df, OUTPUT_DIR)

    # ── Save summary ───────────────────────────────────────────────────────
    save_summary(long_df, OUTPUT_DIR)

    print("\n" + "="*60)
    print("DONE")
    print("="*60)
    print(f"\nCopy results to local:")
    print(f"  scp -r e20378@ada:{OUTPUT_DIR} ./longitudinal_results")


if __name__ == "__main__":
    main()