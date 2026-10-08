#!/usr/bin/env python
"""Regenerate the Results block of README.md from the committed output CSVs.

Every number between the results markers in README.md is written by this script from files
under modules/*/outputs*/ and modules/story/. The held-out force errors carry a 95 percent
bootstrap interval over the test cells (resampling cells, 4000 draws, seed 0), so differences
between runs can be read against the scatter of a 20- or 25-cell test set.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # so the script also runs without `pip install -e .`
    sys.path.insert(0, str(ROOT))
from casebook.metrics import bootstrap_rmse  # noqa: E402  (the notebook uses the same function)
README = ROOT / "README.md"
START, END = "<!-- results:start -->", "<!-- results:end -->"
M0 = ROOT / "modules/M0_foundation_triage"
M1 = ROOT / "modules/M1_finetune_asio2/outputs"

# run folder (or "" for arm A at the top of outputs/) -> (label, held-out set, training set)
M1_RUNS = [
    ("", "naive, seed 0", "20 random silica cells", "51 silica cells"),
    ("am26_asio2_mpa0_naive_s1", "naive, seed 1", "same 20 cells", "same 51 cells"),
    ("am26_asio2_mpa0_naive_s2", "naive, seed 2", "same 20 cells", "same 51 cells"),
    ("am26_asio2_mpa0_naive_holdmin", "naive, repeat", "same 20 cells", "same 51 cells (same split as seed 0)"),
    ("am26_asio2_mpa0_naive_n40", "naive, learning curve", "same 20 cells", "40 of the 51"),
    ("am26_asio2_mpa0_naive_n20", "naive, learning curve", "same 20 cells", "20 of the 51"),
    ("am26_asio2_mpa0_naive_n10", "naive, learning curve", "same 20 cells", "10 of the 51"),
    ("am26_asio2_mpa0_naive_noisy10", "naive, corrupted labels", "same 20 cells", "51, 5 with noisy labels"),
    ("am26_asio2_mpa0_replay_short", "multihead replay, 30 epochs", "same 20 cells", "51 + 1000 replay"),
    ("am26_asio2_mpa0_naive_rate14", "naive, rate hold-out", "20 cells at 10¹⁴ K/s", "51 at 10¹¹–10¹³ K/s"),
    ("am26_asio2_mpa0_naive_rate11", "naive, rate hold-out", "20 cells at 10¹¹ K/s", "51 at 10¹²–10¹⁴ K/s"),
    ("am26_aC_mpa0_naive", "naive, carbon", "25 cells at 1.5–2.0 g/cm³", "64 carbon cells at 2.0–3.5 g/cm³"),
]


def md_table(df: pd.DataFrame, cols: list[str], fmt: dict[str, str], rename: dict[str, str] | None = None) -> str:
    cols = [c for c in cols if c in df.columns]
    head = [(rename or {}).get(c, c) for c in cols]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt.get(c, "{}").format(r[c]) if pd.notna(r[c]) else "" for c in cols) + " |")
    return "\n".join(lines)


def boot_rmse(per_frame: Path, seed: int = 0, draws: int = 4000) -> tuple[float, float, float]:
    fr = pd.read_csv(per_frame)
    return bootstrap_rmse(fr["frame_force_rmse_meV_A"], fr["natoms"], seed=seed, draws=draws)


def m1_table() -> str | None:
    rows = []
    for folder, label, test, train in M1_RUNS:
        d = M1 / folder if folder else M1
        csv = d / "heldout_metrics.csv"
        if not csv.exists():
            continue
        m = pd.read_csv(csv)
        zs, ft = m[m["model"] == "zero-shot"].iloc[0], m[m["model"] != "zero-shot"].iloc[0]
        pf_ft = d / f"heldout_per_frame_{ft['model']}.csv"
        pf_zs = d / "heldout_per_frame_zero-shot.csv"
        ci_ft = boot_rmse(pf_ft) if pf_ft.exists() else (ft["force_rmse_meV_A"], np.nan, np.nan)
        ci_zs = boot_rmse(pf_zs) if pf_zs.exists() else (zs["force_rmse_meV_A"], np.nan, np.nan)
        rows.append({"run": label, "held-out set": test, "trained on": train,
                     "zero-shot F": f"{ci_zs[0]:.1f} [{ci_zs[1]:.0f}–{ci_zs[2]:.0f}]",
                     "fine-tuned F": f"{ci_ft[0]:.1f} [{ci_ft[1]:.1f}–{ci_ft[2]:.1f}]",
                     "zero-shot E": zs["energy_mae_meV_atom"], "fine-tuned E": ft["energy_mae_meV_atom"]})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    return ("**M1 held-out errors.** Force RMSE per Cartesian component in meV/Å with a 95 % bootstrap interval "
            "over the test cells; energy MAE in meV/atom. Every run starts from MACE-MPA-0. The first nine rows share "
            "one random split, so their differences can be compared directly; the rate and carbon rows use other test sets. "
            "Sources: `modules/M1_finetune_asio2/outputs/*/heldout_metrics.csv` and the per-frame files beside them.\n\n"
            + md_table(df, list(df.columns), {"zero-shot E": "{:.2f}", "fine-tuned E": "{:.2f}"}))


def m0_section(title: str, folder: Path, note: str) -> str | None:
    """M0 metrics, plus the energy MAE left after removing each model's mean offset on that
    system (from the per-frame files) and the Spearman coefficient from run_summary.json."""
    path = folder / "metrics.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    offset, mae_rel = [], []
    for _, r in df.iterrows():
        pf = folder / f"per_frame_{r['system']}_{r['model']}.csv"
        if pf.exists():
            e = pd.read_csv(pf)["energy_error_meV_atom"]
            offset.append(e.mean())
            mae_rel.append((e - e.mean()).abs().mean())
        else:
            offset.append(np.nan)
            mae_rel.append(np.nan)
    df["energy_offset_meV_atom"] = offset
    df["energy_mae_offset_removed"] = mae_rel
    cols = ["system", "model", "licence", "n_frames", "force_rmse_meV_A", "force_rel_rmse_percent", "energy_mae_meV_atom",
            "energy_offset_meV_atom", "energy_mae_offset_removed", "sec_per_frame"]
    fmt = {"force_rmse_meV_A": "{:.1f}", "force_rel_rmse_percent": "{:.1f}", "energy_mae_meV_atom": "{:.2f}",
           "energy_offset_meV_atom": "{:.1f}", "energy_mae_offset_removed": "{:.2f}", "sec_per_frame": "{:.2f}"}
    rename = {"force_rmse_meV_A": "force RMSE", "force_rel_rmse_percent": "relative RMSE, %", "energy_mae_meV_atom": "energy MAE",
              "energy_offset_meV_atom": "mean energy offset", "energy_mae_offset_removed": "energy MAE, offset removed",
              "sec_per_frame": "s per frame"}
    rho = ""
    summ = folder / "run_summary.json"
    if summ.exists():
        systems = json.loads(summ.read_text(encoding="utf-8")).get("systems", {})
        vals = [f"{k} {v['spearman_rho_spread_vs_error']:.2f}" for k, v in systems.items() if "spearman_rho_spread_vs_error" in v]
        if vals:
            rho = (" Spearman coefficient between the force spread of the models and the error of their mean, over all atoms: "
                   + ", ".join(vals) + ".")
    rel = path.relative_to(ROOT).as_posix()
    return f"**{title}** (`{rel}`) {note}{rho}\n\n" + md_table(df, cols, fmt, rename)


def section(title: str, path: Path, cols, fmt, rename=None, note: str = "") -> str | None:
    if not path.exists():
        return None
    rel = path.relative_to(ROOT).as_posix()
    return f"**{title}** (`{rel}`){(' ' + note) if note else ''}\n\n" + md_table(pd.read_csv(path), cols, fmt, rename)


def main() -> int:
    parts: list[str] = []
    s = m0_section("M0 zero-shot, silica and carbon", M0 / "outputs",
                   note="Force RMSE per Cartesian component in meV/Å; relative RMSE in percent of the standard deviation of the "
                        "reference force components; energies in meV/atom, the offset being the mean of predicted minus reference "
                        "over the system's cells; seconds per frame on 8 CPU cores.")
    if s:
        parts.append(s)
    s = m0_section("M0 zero-shot, the other AM26 systems, MIT models only", M0 / "outputs_other_systems",
                   note="Systems are labelled by composition, so AM26's binary end-member cells (GeTe, Sb–Te, Li–S, P–S; "
                        "120 frames) are not in these rows.")
    if s:
        parts.append(s)

    s = m1_table()
    if s:
        parts.append(s)

    md_fmt = {"ps_completed": "{:.0f}", "epot_drift_second_half_meV_atom_per_ps": "{:.2f}", "dmin_start_A": "{:.2f}", "dmin_end_A": "{:.2f}",
              "frac_Si_CN4_end": "{:.3f}", "ms_per_step": "{:.0f}", "T_K": "{:.0f}"}
    md_cols = ["model", "T_K", "ps_completed", "finished", "epot_drift_second_half_meV_atom_per_ps", "dmin_start_A", "dmin_end_A", "frac_Si_CN4_end", "ms_per_step"]
    for d, note in ((M1 / "md_stability", "float64, zero-shot and fine-tuned"), (M1 / "md_stability_am26_asio2_mpa0_naive", "float32, fine-tuned")):
        s = section(f"M1 MD check, 20 ps Langevin NVT on one held-out silica cell, {note}", d / "md_stability_summary.csv", md_cols, md_fmt,
                    rename={"epot_drift_second_half_meV_atom_per_ps": "drift, meV/atom/ps", "frac_Si_CN4_end": "Si CN4 at end"})
        if s:
            parts.append(s)

    mq = M1 / "melt_quench_5000K"
    s = section("M1 melt-quench, 5000 K for 20 ps then 10¹⁴ K/s to 300 K, fine-tuned model: the model's glass against AM26's 10¹⁴ K/s cells",
                mq / "melt_quench_vs_am26.csv", ["descriptor", "n_generated", "generated_mean", "generated_sd", "n_am26", "am26_mean", "am26_sd"],
                {"generated_mean": "{:.3f}", "generated_sd": "{:.3f}", "am26_mean": "{:.3f}", "am26_sd": "{:.3f}"},
                note="Density is set by the starting cell (fixed volume) and is not a result.")
    if s:
        parts.append(s)
    s = section("M1 melt-quench at 5000 K, per run", mq / "melt_quench_summary.csv",
                ["seed", "start_label", "finished", "ms_per_step", "msd_Si_end_hold_A2", "ell2_A2", "bridges_kept_end_hold", "network_reset", "dmin_end_A", "frac_Si_CN4_end"],
                {"ms_per_step": "{:.0f}", "msd_Si_end_hold_A2": "{:.0f}", "ell2_A2": "{:.1f}", "bridges_kept_end_hold": "{:.3f}", "dmin_end_A": "{:.2f}", "frac_Si_CN4_end": "{:.3f}"},
                rename={"seed": "run", "msd_Si_end_hold_A2": "Si MSD end of hold, Å²", "ell2_A2": "(V/N)^(2/3), Å²", "bridges_kept_end_hold": "bridges kept"})
    if s:
        parts.append(s)
    parts.append("The first melt-quench attempt, 10 ps at 3000 K, did not melt the network (93 to 100 percent of Si–O–Si bridges "
                 "survived); its outputs stay in `melt_quench_naive/`, `melt_quench_zeroshot/` and `melt_quench_traj/` for the record "
                 "and are described in `modules/M1_finetune_asio2/README.md`.")

    s = section("M2 support criterion per quench rate (silica) and density (carbon), q_conv = 0.2",
                ROOT / "modules/M2_ensemble_support/outputs/support_summary.csv",
                ["system", "group", "descriptor", "N_available", "R_at_N", "reached_q_conv", "n_required", "n_extrapolated_sqrt"],
                {"R_at_N": "{:.2f}", "n_extrapolated_sqrt": "{:.0f}", "n_required": "{:.0f}"},
                rename={"n_extrapolated_sqrt": "n needed (1/√n estimate)"},
                note="For carbon, density is the grouping variable, so its R = 0 is true by construction.")
    if s:
        parts.append(s)

    s = section("M4 net force of the AM26 reference labels, |ΣF|/N in meV/Å", ROOT / "modules/M4_label_quality/outputs/net_force_summary.csv",
                ["system", "n_frames", "median_meV_A", "p95_meV_A", "max_meV_A", "fraction_above_1_meV_A"],
                {"median_meV_A": "{:.1e}", "p95_meV_A": "{:.1e}", "max_meV_A": "{:.1e}", "fraction_above_1_meV_A": "{:.2f}"},
                note="Rows marked `unknown:` are AM26's binary end-member cells, labelled by composition.")
    if s:
        parts.append(s)

    body = "\n\n".join(parts) if parts else "No results yet."
    text = README.read_text(encoding="utf-8")
    i, j = text.index(START) + len(START), text.index(END)
    README.write_text(text[:i] + "\n" + body + "\n" + text[j:], encoding="utf-8")
    print(f"[results] README updated with {len(parts)} block(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
