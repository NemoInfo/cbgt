import marimo

__generated_with = "0.11.20"
app = marimo.App()


@app.cell
def _():
  import marimo as mo
  import numpy as np
  import matplotlib.pyplot as plt
  import matplotlib as mpl
  from scipy.signal import convolve

  mpl.rcParams.update({
      "font.family": "serif",
      "font.serif": ["Libertinus Serif"],
      "font.sans-serif": ["Libertinus Sans"],
      "font.size": 11,                        # base font size in pt for labels/ticks
      "axes.titlesize": 15,
      "axes.labelsize": 12,
      "xtick.labelsize": 10,
      "ytick.labelsize": 10,
      "legend.fontsize": 10,
      "pdf.fonttype": 42,                     # 42 = TrueType (keeps text as text)
      "ps.fonttype": 42,
  })
  return convolve, mo, mpl, np, plt


@app.cell
def _(convolve, np, plt):

  def p_si_given_sj(si, sj, sig_s=0.1):
    return 1 / (1 + ((si - sj) / sig_s)**2)

  def rayleigh_kernel(sigma, dt, t_max):
    t = np.arange(0, t_max, dt)
    kernel = (t / sigma**2) * np.exp(-t**2 / (2 * sigma**2))
    return kernel / np.sum(kernel)

  def convolve_spike_train(spike_train, kernel):
    return np.array([
        convolve(spike_train[:, i], kernel, mode='full')[:spike_train.shape[0]] for i in range(spike_train.shape[1])
    ]).T

  def enforce_min_isi(spikes, dt, min_isi_ms=2.0):
    _, N = spikes.shape
    min_isi_steps = int(min_isi_ms / dt)
    cleaned_spikes = np.zeros_like(spikes)

    for i in range(N):
      spike_times = np.where(spikes[:, i])[0]
      if spike_times.size == 0:
        continue

      last_spike = spike_times[0]
      cleaned_spikes[last_spike, i] = 1

      for t in spike_times[1:]:
        if t - last_spike >= min_isi_steps:
          cleaned_spikes[t, i] = 1
          last_spike = t

    return cleaned_spikes

  plt.rcParams.update({'font.size': 14})
  return (
      convolve_spike_train,
      enforce_min_isi,
      p_si_given_sj,
      rayleigh_kernel,
  )


@app.cell
def _(enforce_min_isi, np, p_si_given_sj, plt):
  N = 50
  r_base = 4 / 1e3  # 4Hz in 1/ms
  r_stim = 40 / 1e3 # 4Hz in 1/ms
  dt = 0.05         # ms
  T = int(4e3 / dt) # 4s
  u = np.random.rand(T, N)
  ts = np.arange(T) * dt
  stims = np.zeros((T, N))
  stims[int(0.2e3 / dt):int(1.2e3 / dt), 24] = 1
  stims[int(3e3 / dt):int(4e3 / dt), 24] = 1
  rates = np.full((T, N), r_base)
  spos = np.linspace(-1, 1, N)
  sx = spos[24]

  for _i in range(N):
    for _j in range(stims.shape[1]):
      rates[:, _i] += stims[:, _j] * p_si_given_sj(spos[_i], spos[_j]) * r_stim

  rates = np.minimum(rates, 40)
  spikes = u < (rates * dt)
  spikes = enforce_min_isi(spikes, dt)

  _fig, _ax = plt.subplots(1, 1, figsize=(8 / 2.54, 8 / 2.54), sharex=True)

  for _i, spike in enumerate(spikes.T):
    # spike times
    t_spikes = ts[spike]
    # y position for this neuron/context
    y_spikes = [spos[_i]] * spike.sum()

    _ax.plot(t_spikes, y_spikes, ".", color="black", markersize=2)

  _ax.set_ylim((-1, 1))
  _ax.set_yticks([-1, 0, 1])
  _fig.suptitle("B", x=0.15)
  # _ax.set_yticklabels([r"$-s^{CTX}_x / L$", "0", r"$s^{CTX}_x / L$"])
  _ax.set_yticklabels(["", "", ""])
  _ax.set_xlim((ts[0], ts[-1]))
  _ax.set_xlabel("time")
  _ax.set_title('CTX spike train')
  _ax.set_xticks(np.arange(0, 4001, 1000))
  _ax.set_xlabel('Time (ms)')

  plt.savefig("figs/ctx_b.svg")
  plt.show()
  return (
      N,
      T,
      dt,
      r_base,
      r_stim,
      rates,
      spike,
      spikes,
      spos,
      stims,
      sx,
      t_spikes,
      ts,
      u,
      y_spikes,
  )


@app.cell
def _(np, p_si_given_sj, plt, sx):
  _fig = plt.figure(figsize=(2 / 2.54, 8 / 2.54))
  ax_profile = _fig.gca()
  fspos = np.linspace(-1, 1, 100)
  ax_profile.set_ylim(-1, 1)
  ax_profile.set_xlim(-0.1, 1.2)
  ax_profile.set_yticks([])
  ax_profile.set_xticks([])
  ax_profile.plot(1 - p_si_given_sj(fspos, sx) * 0.5, fspos, 'k')

  for spine in ax_profile.spines.values():
    spine.set_visible(False)
  ax_profile.set_ylabel("$s_x$", rotation=0, labelpad=10)
  ax_profile.yaxis.set_label_position("right")
  _fig.suptitle("A", x=0)
  ax_profile.set_title("stimulus", loc="left", x=0.2)
  ax_profile.set_xlabel("$p(s^{CTX}_x|s_j)$")
  ax_profile.annotate('',
                      xy=(1.2, 1),
                      xytext=(1.2, -1),
                      arrowprops=dict(arrowstyle='<->', color='black', lw=1.5),
                      xycoords='data',
                      textcoords='data')

  plt.savefig("figs/ctx_a.svg")
  plt.show()
  return ax_profile, fspos, spine


@app.cell
def _(convolve_spike_train, dt, np, plt, rayleigh_kernel, spikes, ts):
  sigma = 8.0
  t_max = 50.0
  kernel = rayleigh_kernel(sigma, dt, t_max)
  syns = convolve_spike_train(spikes, kernel)

  ids = [0, 23, 24, 25, -1]
  aids = [0, 3, 4, 5, -1]
  keep = 10
  fig, axs = plt.subplots(9, 1, figsize=(8 / 2.54, 8 / 2.54), sharex=False, sharey=True)
  for i, (ax, syn) in enumerate(zip(axs[aids], syns.T[ids])):
    ax.plot(ts, syn, color='black', lw=0.5)
    ax.yaxis.set_label_position("right")
    ax.set_xlim((ts[0], ts[-1]))
    ax.set_ylim((-0.0005, 0.007))
    ax.set_yticks([])
    ax.set_xticks([])
    # ax.set_ylabel(f"CTX {i+1}", rotation=0, labelpad=30)

  for ax in axs:
    if ax not in axs[aids]:
      ax.set_visible(False)

  fig.suptitle("C", x=0.15)
  axs[0].set_title("CTX synaptic ouput")
  axs[-1].set_xticks(np.arange(0, 4001, 1000))
  axs[-1].set_xlabel('Time (ms)')

  plt.savefig("figs/ctx_c.svg")
  plt.show()
  return aids, ax, axs, fig, i, ids, keep, kernel, sigma, syn, syns, t_max


@app.cell
def _(np, plt):
  from scipy.stats import rayleigh
  _fspos = np.linspace(0, 3, 200)        # x values
  _pdf = rayleigh.pdf(_fspos, scale=0.5) # Rayleigh distribution

  _fig = plt.figure(figsize=(2 / 2.54, 2 / 2.54))
  _ax_profile = _fig.gca()

  _ax_profile.set_ylim((-0.2, np.max(_pdf) + 0.05))
  _ax_profile.set_yticks([])
  _ax_profile.set_xticks([])

  # horizontal style: pdf along x-axis, variable on y-axis
  _ax_profile.plot(_fspos, _pdf, 'k')

  # remove spines
  for _spine in _ax_profile.spines.values():
    _spine.set_visible(False)

  _ax_profile.set_ylabel(r"$\ast_t$", rotation=0, labelpad=10)
  _ax_profile.set_xlabel(r"$t$", loc="right")
  _ax_profile.annotate('',
                       xy=(-0.15, -0.2),
                       xytext=(3.3, -0.2),
                       arrowprops=dict(arrowstyle='<-', color='black', lw=1.5),
                       xycoords='data',
                       textcoords='data')

  plt.savefig("figs/ctx_supplementary.svg")
  plt.show()
  return (rayleigh, )


if __name__ == "__main__":
  app.run()
