import matplotlib.pyplot as plt
import numpy as np
from typing import Any
from matplotlib.ticker import FormatStrFormatter


def plot_time_activity(dfs, labels, title=None, cmap="gray_r", file=None, y="v", unit="", vmax=1, vmin=0):
  assert len(dfs) != 0
  _vmin, _vmax = vmin, vmax

  unit = f"({unit})" if unit != "" else unit
  dt = dfs[0]["time"][1]

  xs = [np.stack(df[y]) for df in dfs if y in df]

  fig, axs = plt.subplots(len(xs), 1, sharex=True, figsize=(8, 10 * 0.3))
  axs = np.atleast_1d(axs)

  im: Any = None
  for ax, x, label in zip(axs, xs, labels):
    vmax = max(_vmax, x.T[:].max())
    vmin = min(_vmin, x.T[:].min())
    im = ax.imshow(x.T[:], aspect='auto', cmap=cmap, interpolation='nearest', vmin=vmin, vmax=vmax)
    ax.set_yticks([0, x.shape[1] - 1])
    ax.set_yticklabels([1, x.shape[1]])
    ax.set_ylabel(f"{label} #", rotation=0, fontsize=14, labelpad=15)
    cbar = fig.colorbar(
        im,
        ax=ax,
        location="right",
        label=f"Voltage {unit}",
        shrink=0.8,
        aspect=10,
    )
    cbar.set_ticks(np.linspace(vmin, vmax, 3))
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    cbar.set_label(unit, rotation=0, fontsize=14, labelpad=15)

  axs[-1].set_xticks([0, xs[-1].shape[0]])
  axs[-1].xaxis.set_major_formatter(lambda x, _: f"{x*dt:.0f}")
  axs[-1].set_xlabel("Time (ms)", fontsize=14)

  if title is None: title = y
  axs[0].set_title(title, fontsize=16)
  if file is not None: plt.savefig("file")


def plot_time_trace(dfs, labels, title=None, file=None, color="k", y="v"):
  dt = dfs[0]["time"][1]
  xs = [np.stack(df[y]) for df in dfs]
  max_num_neurons = max([x.shape[1] for x in xs])
  fig, axs = plt.subplots(max_num_neurons,
                          len(xs),
                          figsize=(4.5 * len(xs), max_num_neurons * 0.6),
                          sharex=True,
                          tight_layout=True,
                          gridspec_kw={'hspace': 0})

  for axcol, x, label in zip(axs.T, xs, labels):
    for i, (trace, ax) in enumerate(zip(x.T, axcol)):
      ax.plot(trace, color=color, lw=1)
      ax.set_yticks([])
      ax.set_ylabel(f"{label} {i+1}", rotation=0, labelpad=10)
      ax.yaxis.set_label_coords(-0.07, 0.3)

  axs[-1, 0].xaxis.set_major_formatter(lambda x, _: f"{x*dt:.0f}")
  fig.supxlabel("Time (ms)")

  if title is not None: fig.suptitle(title, fontsize=16)
  if file is not None: plt.savefig(file)
