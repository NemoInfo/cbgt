import marimo

__generated_with = "0.11.20"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import cbgt
    from src.plot import plot_time_activity, plot_time_trace
    import scipy.signal
    import polars as pl

    plt.style.use('dark_background')
    plt.rcParams["figure.facecolor"] = "#101418"
    plt.rcParams["axes.facecolor"] = "#101418"
    cmap="YlGnBu"
    return (
        cbgt,
        cmap,
        mo,
        mpl,
        np,
        pl,
        plot_time_activity,
        plot_time_trace,
        plt,
        scipy,
    )


@app.cell
def _(cbgt):
    rt = cbgt.Network(dt=0.05, total_t=6, experiment="cluster", stn_ca_pre=0, stn_ca_post=0)
    d_stn, d_gpe, d_gpi = rt.run_rk4()
    df = rt.to_polars(0.05)
    return d_gpe, d_gpi, d_stn, df, rt


@app.cell
def _(np):
    def spike_train_to_lfp(spike_train, dt, rayleigh_scale=0.05e3, kernel_duration=0.3e3):
      from scipy.stats import rayleigh
      num_time_steps, num_neurons = spike_train.shape

      num_kernel_points = int(kernel_duration / dt)
      if num_kernel_points == 0:
          num_kernel_points = 1
      x = np.arange(0, num_kernel_points * dt, dt)

      kernel = rayleigh.pdf(x, scale=rayleigh_scale)
      kernel = kernel / np.sum(kernel) # Normalize kernel
      lfp_signals = np.zeros_like(spike_train, dtype=float)
      for i in range(num_neurons):
          lfp_signals[:, i] = np.convolve(spike_train[:, i], kernel, mode='same')

      return lfp_signals
    return (spike_train_to_lfp,)


@app.cell
def _(d_gpi, d_stn, plt, spike_train_to_lfp):
    lfp_stn = spike_train_to_lfp(d_gpi, 0.05, rayleigh_scale=50, kernel_duration=300)

    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(15, 2))
    a1.plot(d_stn[:, 0])
    a2.plot(lfp_stn[:, 0])
    a3.plot(lfp_stn[:].sum(axis=1))
    #a3.ylim(0, )
    plt.show()
    return a1, a2, a3, fig, lfp_stn


@app.cell
def _(lfp_stn, plt):
    from scipy.signal import welch

    freqs, stn_psd = welch(lfp_stn[:,[2]].sum(axis=1), fs=1e3/0.05, nperseg=1e3/0.05)
    stn_psd = stn_psd / stn_psd.sum()

    plt.figure(figsize=(10, 5))
    plt.plot(freqs, stn_psd, label='STN')
    # plt.yscale("log")
    plt.xlim(0, 30)
    plt.legend()
    plt.show()
    return freqs, stn_psd, welch


if __name__ == "__main__":
    app.run()
