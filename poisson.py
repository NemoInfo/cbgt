import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve


def p_si_given_sj(si, sj, sig_s=0.1):
  return 1 / (1 + ((si - sj) / sig_s)**2)


def rayleigh_kernel(sigma, dt, t_max):
  t = np.arange(0, t_max, dt)
  kernel = (t / sigma**2) * np.exp(-t**2 / (2 * sigma**2))
  return kernel / np.sum(kernel)


def convolve_spike_train(spike_train, kernel):
  return np.array(
      [convolve(spike_train[:, i], kernel, mode='full')[:spike_train.shape[0]] for i in range(spike_train.shape[1])]).T


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


def main():
  plt.rcParams.update({'font.size': 12})

  N = 10
  r_base = 4 / 1e3  # 4Hz in 1/ms
  r_stim = 40 / 1e3 # 4Hz in 1/ms
  dt = 0.05         # ms
  T = int(4e3 / dt) # 4s
  u = np.random.rand(T, N)
  ts = np.arange(T) * dt
  stims = np.zeros((T, N))
  stims[int(0.2e3 / dt):int(1.2e3 / dt), 4] = 1
  stims[int(3e3 / dt):int(4e3 / dt), 4] = 1
  rates = np.full((T, N), r_base)
  spos = np.linspace(0, 1, N)
  sx = spos[5]

  for i in range(N):
    for j in range(stims.shape[1]):
      rates[:, i] += stims[:, j] * p_si_given_sj(spos[i], spos[j]) * r_stim

  rates = np.minimum(rates, 40)
  spikes = u < (rates * dt)
  spikes = enforce_min_isi(spikes, dt)

  fig, axs = plt.subplots(N, 1, figsize=(10, 1 * N), sharex=True)
  for i, (ax, spike) in enumerate(zip(axs, spikes.T)):
    ax.vlines(ts[spike], 0, 1, color='black') # vertical lines where spikes occur
    ax.fill_between(ts, 0, 1, where=stims[:, i] > 0, color="gray", alpha=0.3)
    ax.yaxis.set_label_position("right")
    ax.set_xlim((ts[0], ts[-1]))
    ax.set_yticks([])
    ax.set_ylabel(f"CTX {i+1}", rotation=0, labelpad=30)

  axs[0].set_title('Poisson Spike Train')
  axs[-1].set_xticks(np.arange(0, 4001, 1000))
  axs[-1].set_xlabel('Time (ms)')

  plt.show(block=False)

  fig = plt.figure(figsize=(2, 10))
  ax_profile = fig.gca()
  spos = np.linspace(0, 1, 100)
  ax_profile.set_ylim(0, 1)
  ax_profile.set_xlim(-0.1, 1.2)
  ax_profile.set_yticks([])
  ax_profile.set_xticks([])
  ax_profile.plot(1 - p_si_given_sj(spos, sx), spos, 'k')

  for spine in ax_profile.spines.values():
    spine.set_visible(False)
  ax_profile.set_ylabel("$s_x$", rotation=0, labelpad=10)
  ax_profile.yaxis.set_label_position("right")
  ax_profile.set_title("Cortical stimuli")
  ax_profile.set_xlabel("$p(s^{CTX}_x|s_j)$")
  ax_profile.annotate('',
                      xy=(1.2, 1),
                      xytext=(1.2, 0),
                      arrowprops=dict(arrowstyle='<->', color='black', lw=1.5),
                      xycoords='data',
                      textcoords='data')

  plt.show(block=False)

  sigma = 5.0
  t_max = 50.0

  kernel = rayleigh_kernel(sigma, dt, t_max)
  syns = convolve_spike_train(spikes, kernel)

  fig, axs = plt.subplots(N, 1, figsize=(10, 1 * N), sharex=True)
  for i, (ax, syn) in enumerate(zip(axs, syns.T)):
    ax.plot(ts, syns[:, i], color='black')
    ax.yaxis.set_label_position("right")
    ax.set_xlim((ts[0], ts[-1]))
    ax.set_yticks([])
    ax.set_ylabel(f"CTX {i+1}", rotation=0, labelpad=30)

  axs[-1].set_xticks(np.arange(0, 4001, 1000))
  axs[-1].set_xlabel('Time (ms)')

  plt.show()


if __name__ == "__main__": main()
