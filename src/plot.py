import matplotlib.pyplot as plt
import numpy as np
from typing import Any


def plot_time_activity(dfs, labels, title=None, vmin=None, vmax=0, cmap="gray_r", file=None, y="v"):
  assert len(dfs) != 0
  dt = dfs[0]["time"][1]
  xs = [np.stack(df[y]) for df in dfs]
  num_neurons = sum([x.shape[1] for x in xs])

  fig, axs = plt.subplots(len(xs), 1, sharex=True, figsize=(8, num_neurons * 0.3))
  axs = np.atleast_1d(axs)

  if vmin is None: vmin = min([x.min() for x in xs])
  if vmax is None: vmax = max([x.max() for x in xs])

  im: Any = None
  for ax, x, label in zip(axs, xs, labels):
    im = ax.imshow(x.T, aspect='auto', cmap=cmap, interpolation='nearest', vmin=vmin, vmax=vmax)
    ax.set_yticks([0, x.shape[1] - 1])
    ax.set_yticklabels([1, x.shape[1]])
    ax.set_ylabel(f"{label} #", rotation=0, fontsize=14, labelpad=15)

  axs[-1].set_xticks([0, xs[-1].shape[0]])
  axs[-1].xaxis.set_major_formatter(lambda x, _: f"{x*dt:.0f}")

  axs[-1].set_xlabel("Time (ms)", fontsize=14)

  cbar = fig.colorbar(im, ax=axs, location="right", label="Voltage (mV)", shrink=0.8, aspect=50)
  cbar.set_ticks(np.arange(vmax, vmin - 1, -20))
  cbar.set_label("mV", rotation=0, fontsize=14, labelpad=15)

  if title is not None: axs[0].set_title(title, fontsize=16)
  if file is not None: plt.savefig("file")

  plt.show()


def plot_time_trace(dfs, labels, title=None, file=None, color="k", y="v"):
  dt = dfs[0]["time"][1]
  xs = [np.stack(df[y]) for df in dfs]
  max_num_neurons = max([x.shape[1] for x in xs])
  fig, axs = plt.subplots(max_num_neurons,
                          len(xs),
                          figsize=(4.5 * len(xs), max_num_neurons * 0.6),
                          sharex=True,
                          sharey=True,
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

  plt.show()
