#!/usr/bin/env python
"""Video of a melt-quench: the atoms, the RDFs and the coordination, frame by frame.

Left: the cell, Si atoms coloured by their O coordination from CRISP's calculate_coordination
(3 orange, 4 red, 5 or more blue), O in light grey. Middle: Si-O and Si-Si partial RDFs from
CRISP's compute_pairwise_rdf for the current frame (solid) against the AM26 10^14 K/s average
(dashed). Right: fraction of Si with 3, 4 and 5 oxygens along the quench, temperature dotted,
a marker at the current time.

  python tools/story_video.py --crisp-src /path/to/CRISP_Project \
      --traj modules/M1_finetune_asio2/outputs/melt_quench_traj/traj_naive_seed0.xyz

Writes modules/story/quench_video.mp4 when ffmpeg is available (the imageio-ffmpeg wheel is
enough), otherwise quench_video.gif, plus quench_video_frame_<k>.png stills for the slide.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from casebook.io import filter_system, group_from, read_frames

ROOT = Path(__file__).resolve().parents[1]
M1 = ROOT / "modules" / "M1_finetune_asio2" / "outputs"
CN_COLOURS = {0: "#4d4d4d", 1: "#4d4d4d", 2: "#4d4d4d", 3: "#e69f00", 4: "#c0392b", 5: "#1f77b4", 6: "#1f77b4"}  # >=7 falls back to blue


def load_module(crisp_src: str, rel: str, name: str):
    path = Path(crisp_src) / "CRISP" / "data_analysis" / rel
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rdf_pair(fn, atoms, a, b, rmax, nbins):
    sym = np.asarray(atoms.get_chemical_symbols())
    ref = [int(k) for k in np.where(sym == a)[0]]
    tgt = [int(k) for k in np.where(sym == b)[0]]
    g, r = fn(atoms, ref, tgt, rmax, nbins)
    return np.asarray(r), np.asarray(g)


def ffmpeg_writer(fps):
    """FFMpegWriter via an ffmpeg on PATH or the imageio-ffmpeg wheel; None if neither exists."""
    import shutil
    import matplotlib
    from matplotlib.animation import FFMpegWriter
    exe = shutil.which("ffmpeg")
    if exe is None:
        try:
            import imageio_ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return None
    matplotlib.rcParams["animation.ffmpeg_path"] = exe
    return FFMpegWriter(fps=fps, codec="libx264", bitrate=2400, extra_args=["-pix_fmt", "yuv420p"])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--crisp-src", required=True)
    p.add_argument("--traj", default=str(M1 / "melt_quench_5000K" / "traj_naive_seed4.xyz"))
    p.add_argument("--am26", default=str(ROOT / "data" / "am26.extxyz"))
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--rotation", default="12x,18y,0z")
    p.add_argument("--stills", type=int, default=3, help="number of still frames to save as PNG")
    p.add_argument("--out", default=str(ROOT / "modules" / "story"))
    a = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.lines import Line2D
    from ase.visualize.plot import plot_atoms
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    prdf = load_module(a.crisp_src, "prdf.py", "crisp_prdf")
    cn = load_module(a.crisp_src, "cn_cal.py", "crisp_cn")
    frames = read_frames(a.traj)
    t = np.array([f.info.get("t_ps", np.nan) for f in frames], dtype=float)
    T = np.array([f.info.get("T_set_K", np.nan) for f in frames], dtype=float)
    n = len(frames)
    print(f"[video] {n} frames")

    silica = filter_system(read_frames(a.am26), "a-SiO2")
    rate = np.array([group_from(c.info.get("label"), r"10-(\d+)") for c in silica], dtype=float)
    ref = [c for c, v in zip(silica, rate) if v == rate.max()]

    pairs = (("Si", "O", 1.2, 3.6, 22), ("Si", "Si", 2.4, 6.0, 5.5))
    nbins = 90
    r_axis, per_frame, ref_mean = {}, {k[:2]: [] for k in pairs}, {}
    for pr in pairs:
        curves = np.array([rdf_pair(prdf.compute_pairwise_rdf, c, pr[0], pr[1], 6.0, nbins)[1] for c in ref])
        ref_mean[pr[:2]] = curves.mean(axis=0)
    cn_per_frame, colours = [], []
    for f in frames:
        dm = f.get_all_distances(mic=True)
        d = cn.calculate_coordination(f, dm, [int(k) for k in range(len(f))], "Si", {("Si", "O"): 2.0})
        cn_per_frame.append(d)
        sym = f.get_chemical_symbols()
        colours.append([CN_COLOURS.get(d.get(i, 4), "#1f77b4") if s == "Si" else "#d9d9d9" for i, s in enumerate(sym)])
        for pr in pairs:
            r, g = rdf_pair(prdf.compute_pairwise_rdf, f, pr[0], pr[1], 6.0, nbins)
            per_frame[pr[:2]].append(g)
            r_axis[pr[:2]] = r
    pct = cn.calculate_avg_percentages(cn_per_frame)  # {CN: [% per frame]}
    pd.DataFrame({"t_ps": t, "T_set_K": T, **{f"Si_CN{k}_percent": v for k, v in pct.items()}}).to_csv(out / "quench_video_coordination.csv", index=False)

    fig = plt.figure(figsize=(13, 5.4))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1.0, 1.1], hspace=0.45, wspace=0.35, left=0.03, right=0.95, top=0.92, bottom=0.17)
    ax_cell = fig.add_subplot(gs[:, 0])
    ax_sio = fig.add_subplot(gs[0, 1])
    ax_sisi = fig.add_subplot(gs[1, 1])
    ax_cn = fig.add_subplot(gs[:, 2])

    for ax, pr in ((ax_sio, pairs[0]), (ax_sisi, pairs[1])):
        ax.plot(r_axis[pr[:2]], ref_mean[pr[:2]], ls="--", color="0.35", lw=1.3)
        ax.set_xlim(pr[2], pr[3]); ax.set_ylim(0, pr[4])
        ax.set_ylabel(f"g({pr[0]}–{pr[1]})")
    ax_sisi.set_xlabel("r (Å)")
    (l_sio,) = ax_sio.plot(r_axis[pairs[0][:2]], per_frame[pairs[0][:2]][0], color="#c0392b", lw=2.2)
    (l_sisi,) = ax_sisi.plot(r_axis[pairs[1][:2]], per_frame[pairs[1][:2]][0], color="#c0392b", lw=2.2)
    ax_sio.legend([Line2D([], [], ls="--", color="0.35"), Line2D([], [], color="#c0392b", lw=2.2)],
                  ["AM26 average, 10$^{14}$ K/s", "this frame"], fontsize=8, frameon=False, loc="center right")

    n_fr = len(t)
    zero = np.zeros(n_fr)
    low = sum((np.asarray(v) for k, v in pct.items() if k <= 2), zero)
    high = sum((np.asarray(v) for k, v in pct.items() if k >= 5), zero)
    for label, series, colour, lw in (("Si with ≤2 O", low, "0.55", 1.0), ("Si with 3 O", np.asarray(pct.get(3, zero)), CN_COLOURS[3], 1.3),
                                      ("Si with 4 O", np.asarray(pct.get(4, zero)), CN_COLOURS[4], 2.0), ("Si with ≥5 O", high, CN_COLOURS[5], 1.0)):
        ax_cn.plot(t, series, color=colour, lw=lw, label=label)
    ax_cn.set_xlabel("time (ps)"); ax_cn.set_ylabel("fraction of Si atoms (%)")
    ax_cn.set_ylim(0, 102)
    ax_T = ax_cn.twinx()
    ax_T.plot(t, T, color="0.5", lw=1, ls=":"); ax_T.set_ylabel("T (K)", color="0.4"); ax_T.set_ylim(0, 1.12 * float(np.nanmax(T)))
    ax_T.spines["top"].set_visible(False)
    marker = ax_cn.axvline(t[0], color="0.2", lw=1)
    ax_cn.legend(fontsize=8, frameon=False, loc="center right")

    def draw_cell(i):
        ax_cell.cla()
        at = frames[i].copy()
        at.wrap()
        plot_atoms(at, ax_cell, radii=0.38, rotation=a.rotation, colors=colours[i], show_unit_cell=2)
        ax_cell.set_axis_off()
        ax_cell.set_title(f"t = {t[i]:5.1f} ps    T = {T[i]:4.0f} K", fontsize=11, loc="left")

    def update(i):
        draw_cell(i)
        l_sio.set_ydata(per_frame[pairs[0][:2]][i])
        l_sisi.set_ydata(per_frame[pairs[1][:2]][i])
        marker.set_xdata([t[i], t[i]])
        return [l_sio, l_sisi, marker]

    fig.text(0.03, 0.03, "Si coloured by O coordination within 2.0 Å: dark grey 2 or fewer, orange 3, red 4, blue 5 or more; O light grey.   Fine-tuned MACE-MPA-0, 10$^{14}$ K/s, NVT.   RDFs and coordination: CRISP.",
             fontsize=8, color="0.3")
    update(0)
    anim = FuncAnimation(fig, update, frames=n, interval=1000 / a.fps, blit=False)
    writer = ffmpeg_writer(a.fps)
    if writer is not None:
        target = out / "quench_video.mp4"
        anim.save(target, writer=writer, dpi=120)
    else:
        target = out / "quench_video.gif"
        anim.save(target, writer=PillowWriter(fps=a.fps), dpi=100)
    print(f"[video] wrote {target}")

    for j, i in enumerate(np.linspace(0, n - 1, a.stills).astype(int)):
        update(i)
        fig.savefig(out / f"quench_video_frame_{j}.png", dpi=150)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
