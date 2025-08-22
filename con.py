import marimo

__generated_with = "0.11.20"
app = marimo.App(width="columns")


@app.cell(column=0)
def _():
  import marimo as mo
  import cbgt
  import numpy as np
  import matplotlib as mpl
  import matplotlib.pyplot as plt
  from src.plot import plot_time_activity, plot_time_trace
  from time import time
  return cbgt, mo, mpl, np, plot_time_activity, plot_time_trace, plt, time


@app.cell
def _(mpl, np):
  target_color = "#5e3c99"
  base = mpl.colors.LinearSegmentedColormap.from_list("custom_purple", ["white", target_color], N=256)
  cmap = mpl.colors.ListedColormap(base(np.linspace(0.25, 1.0, 512)))
  return base, cmap, target_color


@app.cell(hide_code=True)
def _(np):

  def rand_gpe(num):
    ca = [
        0.009931221391373412, 0.01105798337560691, 0.008618198042586294, 0.007842254082952149, 0.01009947181199855,
        0.01439779182000786, 0.01499734401485086, 0.01153249037272892, 0.01131645351818554, 0.01090747666531076
    ]
    h = [
        0.6941693982604106, 0.6960525186649159, 0.6922676967009093, 0.6908829096175084, 0.694450894176868,
        0.7008187424839692, 0.701701517982433, 0.6967207207137638, 0.6962284698338927, 0.6955334306358607
    ]
    n = [
        0.2185706578168535, 0.2172726252685865, 0.2198825985895934, 0.2208386365344371, 0.218376554433867,
        0.2139920161919114, 0.2133851309251945, 0.2168122919020363, 0.2171513971203313, 0.2176303246979667
    ]
    r = [
        0.2573659658746555, 0.267633438957204, 0.2467211480536698, 0.2393102961633564, 0.25894821881624,
        0.29699756425809, 0.3026979450078091, 0.2718222561955378, 0.2692289296543874, 0.265227594196562
    ]
    s = [
        0.000005005384170945523, 0.000004742408605511108, 0.000005285062432830586, 0.000005496855711727898,
        0.000004965314879070837, 0.000004134036862962592, 0.000004035693901984886, 0.000004652783831094474,
        0.000004718489668449042, 0.000004813260429223516
    ]
    v = [
        -67.82599080345415, -67.93189025010138, -67.71998331113508, -67.642675227932, -67.84176237771105,
        -68.20005240297162, -68.25008741948682, -67.96902996444675, -67.94128590225782, -67.90220199361714
    ]
    return {
        "gpe_ca": np.random.choice(ca, size=num, replace=True),
        "gpe_h": np.random.choice(h, size=num, replace=True),
        "gpe_n": np.random.choice(n, size=num, replace=True),
        "gpe_r": np.random.choice(r, size=num, replace=True),
        "gpe_s": np.random.choice(s, size=num, replace=True),
        "gpe_v": np.random.choice(v, size=num, replace=True),
    }

  def rand_gpi(num):
    ca = [
        0.009931221391373412, 0.01105798337560691, 0.008618198042586294, 0.007842254082952149, 0.01009947181199855,
        0.01439779182000786, 0.01499734401485086, 0.01153249037272892, 0.01131645351818554, 0.01090747666531076
    ]
    h = [
        0.6941693982604106, 0.6960525186649159, 0.6922676967009093, 0.6908829096175084, 0.694450894176868,
        0.7008187424839692, 0.701701517982433, 0.6967207207137638, 0.6962284698338927, 0.6955334306358607
    ]
    n = [
        0.2185706578168535, 0.2172726252685865, 0.2198825985895934, 0.2208386365344371, 0.218376554433867,
        0.2139920161919114, 0.2133851309251945, 0.2168122919020363, 0.2171513971203313, 0.2176303246979667
    ]
    r = [
        0.2573659658746555, 0.267633438957204, 0.2467211480536698, 0.2393102961633564, 0.25894821881624,
        0.29699756425809, 0.3026979450078091, 0.2718222561955378, 0.2692289296543874, 0.265227594196562
    ]
    s = [
        0.000005005384170945523, 0.000004742408605511108, 0.000005285062432830586, 0.000005496855711727898,
        0.000004965314879070837, 0.000004134036862962592, 0.000004035693901984886, 0.000004652783831094474,
        0.000004718489668449042, 0.000004813260429223516
    ]
    v = [
        -67.82599080345415, -67.93189025010138, -67.71998331113508, -67.642675227932, -67.84176237771105,
        -68.20005240297162, -68.25008741948682, -67.96902996444675, -67.94128590225782, -67.90220199361714
    ]
    return {
        "gpi_ca": np.random.choice(ca, size=num, replace=True),
        "gpi_h": np.random.choice(h, size=num, replace=True),
        "gpi_n": np.random.choice(n, size=num, replace=True),
        "gpi_r": np.random.choice(r, size=num, replace=True),
        "gpi_s": np.random.choice(s, size=num, replace=True),
        "gpi_v": np.random.choice(v, size=num, replace=True),
    }

  def rand_stn(num):
    ca = [
        0.2994323366425385, 0.4076730264403847, 0.3271760563827424, 0.2456039126383157, 0.3090126869287847,
        0.3533066857313201, 0.3668697913124569, 0.3777575381495549, 0.3008309498107221, 0.2631312497961643
    ]
    h = [
        0.5063486245631907, 0.2933274739456392, 0.4828268896903307, 0.5957938758715363, 0.4801708406464686,
        0.397555659151211, 0.3761635970127477, 0.3316364917935809, 0.4881964058107033, 0.5373898124788108
    ]
    n = [
        0.0301468039831072, 0.04412485475791555, 0.02936940165051648, 0.03307223867110721, 0.02961425249063069,
        0.02990618866753074, 0.03096707115136645, 0.03603641291454053, 0.02983123244237023, 0.03137696787429014
    ]
    r = [
        0.0295473069771012, 0.07318677802595788, 0.03401991571903244, 0.01899268957583912, 0.0322092810112401,
        0.04490215539151968, 0.0496024428039565, 0.05982606979469521, 0.03078507359379932, 0.02403333448524015
    ]
    s = [
        0.008821617722180833, 0.007400276913597601, 0.00850582621763913, 0.009886276645187469, 0.00862235586166425,
        0.00800161199265862, 0.007851916739337694, 0.007654426383227644, 0.008720434017133022, 0.009298664650592724
    ]
    v = [
        -59.62828421888404, -61.0485669306943, -59.9232859246653, -58.70506521874258, -59.81316532105502,
        -60.41737514151719, -60.57000688576042, -60.77581472006873, -59.72163362685856, -59.20177081754847
    ]

    return {
        "stn_ca": np.random.choice(ca, size=num, replace=True),
        "stn_h": np.random.choice(h, size=num, replace=True),
        "stn_n": np.random.choice(n, size=num, replace=True),
        "stn_r": np.random.choice(r, size=num, replace=True),
        "stn_s": np.random.choice(s, size=num, replace=True),
        "stn_v": np.random.choice(v, size=num, replace=True),
    }

  return rand_gpe, rand_gpi, rand_stn


@app.cell
def _(np):

  def w_cluster(stn_count, gpe_count, gpi_count):
    _base = np.zeros(gpe_count)
    _base[[1, -1]] = 1
    gpe_w_g_g = np.vstack([np.roll(_base, i) for i in range(gpe_count)])
    gpe_w_s_g = np.eye(stn_count, M=gpe_count)
    _base = np.zeros(stn_count)
    _base[[2, -2]] = 1
    stn_w_gpe = np.vstack([np.roll(_base, i) for i in range(gpe_count)])
    return {"gpe_w_g_g": gpe_w_g_g, "gpe_w_s_g": gpe_w_s_g, "stn_w_gpe": stn_w_gpe}

  def w_wave(stn_count, gpe_count, gpi_count):
    gpe_w_g_g = 1 - np.eye(gpe_count)
    _base = np.zeros(gpe_count)
    _base[[0, 1, -1]] = 1
    gpe_w_s_g = np.vstack([np.roll(_base, i) for i in range(stn_count)])
    _base = np.zeros(stn_count)
    _base[[-2, -1, 0, 1, 2]] = 1
    stn_w_gpe = np.vstack([np.roll(_base, i) for i in range(gpe_count)])
    return {
        "gpe_w_g_g": gpe_w_g_g,
        "gpe_w_s_g": gpe_w_s_g,
        "stn_w_gpe": stn_w_gpe,
        "gpi_w_g_g": gpe_w_g_g,
        "gpi_w_s_g": gpe_w_s_g
    }

  def w_n_net(stn_count, gpe_count, gpi_count):
    ggn = 130 // 2
    base = np.zeros(gpe_count)
    base[:3] = 1
    gpe_w_s_g = np.vstack([np.roll(base, i * 3) for i in range(stn_count)])
    stn_w_gpe = np.zeros((gpe_count, stn_count))
    stn_w_gpe[np.arange(1, gpe_count, 3), np.arange(stn_count)] = 1
    gpe_w_g_g = sum([np.eye(gpe_count, k=k) for k in list(range(-ggn, 0)) + list(range(1, ggn + 1))])
    return {
        "gpe_w_g_g": gpe_w_g_g,
        "gpe_w_s_g": gpe_w_s_g,
        "stn_w_gpe": stn_w_gpe,
        "gpi_w_g_g": gpe_w_g_g,
        "gpi_w_s_g": gpe_w_s_g
    }

  def white_noise_stn(t, n):
    import numpy as np

    def oscillatory(time_ms, n):
      start, end = 500, 3500
      if time_ms < start or time_ms > end:
        return 0.0

      period_ms = 1000.0 / 10.0
      time_in_cycle = (time_ms - start) % period_ms
      if time_in_cycle < (period_ms / 2):
        return -50.0
      else:
        return 0.0

    return np.random.normal(loc=-20, scale=4) # + oscillatory(t, n)

  def gpe_i_ext(t, n):
    import numpy as np

    def oscillatory(time_ms, n):
      start, end = 500, 3500
      if time_ms < start or time_ms > end:
        return 0.0

      period_ms = 1000.0 / 10.0
      time_in_cycle = (time_ms - start + (period_ms / 2)) % period_ms
      if time_in_cycle < (period_ms / 2):
        return -50.0
      else:
        return 0.0

    return oscillatory(t, n)

  def gpe_i_app(t, n):
    return -.6

  def zeros(t, n):
    return 0.0

  return (
      gpe_i_app,
      gpe_i_ext,
      w_cluster,
      w_n_net,
      w_wave,
      white_noise_stn,
      zeros,
  )


@app.cell
def _(
    cbgt,
    gpe_i_app,
    np,
    rand_gpe,
    rand_gpi,
    rand_stn,
    time,
    w_n_net,
    w_wave,
    white_noise_stn,
):
  np.random.seed(69)
  stn_count, gpe_count, gpi_count = 10, 30, 30

  _map = w_wave(stn_count, gpe_count, gpi_count)

  model = cbgt.Network(
      dt=0.05,
      total_t=8.,
      experiment="cluster",
      gpe_i_app=gpe_i_app,
                                                  #   gpe_i_ext=gpe_i_ext,
                                                  #   stn_ca_pre=0,
                                                  #   stn_ca_post=0,
      stn_theta_d=0.01,
      stn_theta_p=0.05,
      stn_i_ext=white_noise_stn,
      stn_count=stn_count,
      gpe_count=gpe_count,
      gpi_count=gpi_count,
                                                  #   **w_cluster(stn_count, gpe_count, gpi_count),
      **w_n_net(stn_count, gpe_count, gpi_count),
      **rand_gpe(gpe_count),
      **rand_gpi(gpi_count),
      **rand_stn(stn_count),
  )
  start = time()
  s_stn1, s_gpe1, s_gpi1 = model.run_rk4()
  print(f"\n> Simulated 8s in {time() - start:.2f}s 🚀\n")
  df = model.to_polars()
  return (
      df,
      gpe_count,
      gpi_count,
      model,
      s_gpe1,
      s_gpi1,
      s_stn1,
      start,
      stn_count,
  )


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

  return (spike_train_to_lfp, )


@app.cell(hide_code=True)
def _(df, df2, plt):
  # plt.imshow(df["gpe"]["w_s_g"][0], cmap='gray', interpolation='nearest')
  # plt.colorbar(label='Connection Strength')
  # plt.title("Connection Matrix Heatmap")
  # plt.ylabel("STN")
  # plt.xlabel("GPe")
  # plt.tight_layout()
  # plt.show()

  plt.imshow(df["stn"]["w_gpe"][0], cmap='gray', interpolation='nearest')
  plt.colorbar(label='Connection Strength')
  plt.ylabel("GPe")
  plt.xlabel("STN")
  plt.tight_layout()
  plt.show()

  plt.imshow(df2["stn"]["w_gpe"][0], cmap='gray', interpolation='nearest')
  plt.colorbar(label='Connection Strength')
  plt.ylabel("GPe")
  plt.xlabel("STN")
  plt.tight_layout()
  plt.show()

  # plt.imshow(df["gpe"]["w_g_g"][0], cmap='gray', interpolation='nearest')
  # plt.colorbar(label='Connection Strength')
  # plt.xlabel("GPe")
  # plt.ylabel("GPe")
  # plt.tight_layout()
  # plt.show()
  return


@app.cell
def _(np, plt):

  def square_wave(t, A=1, f=600):
    T = 1 / f
    return A * ((t % T) < (T / 2))

  t = np.linspace(0, 0.02, 10000)[:-1000] # 20 ms window
  I1 = square_wave(t)
  I2 = -0.2 - square_wave(t)

  plt.figure(figsize=(4, 2))
  plt.plot(t * 1e3, I1, color="k", lw=2.5, drawstyle='steps-post')
  plt.plot(t * 1e3, I2, color="k", lw=2.5, drawstyle='steps-post')
  plt.grid(False)
  plt.axis("off")
  plt.savefig("stim.svg", transparent=True)
  plt.show()
  return I1, I2, square_wave, t


@app.cell(column=1, hide_code=True)
def _(cmap, df, plot_time_activity, plt):
  plot_time_activity([*df.values()][:-1], ["STN", "GPe", "GPi"], y="v", vmin=-80, vmax=0, cmap=cmap)
  _fig = plt.gcf()
  _fig.set_size_inches(12, 5)
  # plt.savefig("wave_nnet.svg", transparent=True)
  plt.show()
  return


@app.cell
def _(cmap, df2, plot_time_activity, plt):
  plot_time_activity([*df2.values()][:-1], ["STN", "GPe", "GPi"], y="v", vmin=-80, vmax=0, cmap=cmap)
  _fig = plt.gcf()
  _fig.set_size_inches(12, 5)
  # plt.savefig("wave_nnet.svg", transparent=True)
  plt.show()
  return


@app.cell
def _(cmap, df, plot_time_activity, plt):
  plot_time_activity([*df.values()][:-1], [*df.keys()][:-1], y="i_ext", cmap=cmap, vmin=-50, vmax=50)
  plt.show()
  return


@app.cell
def _():
  # fig, (a1, a2) = plt.subplots(2, 1, figsize=(6, 2), sharex=True)
  # a1.plot(lfp_stn_avg[:-5000], lw=2, color="k")
  # a2.plot(lfp_gpe_avg[:-5000], lw=2, color=cmap(1.))
  # a1.set_xlim(0, len(lfp_gpe_avg[:-5000]))
  # a1.set_ylabel("STN", fontsize=14, rotation=0, labelpad=20)
  # a2.set_ylabel("GPe", fontsize=14, rotation=0, labelpad=20)
  # a1.set_xticks([])
  # a1.set_xticklabels([])
  # a1.set_yticks([])
  # a1.set_yticklabels([])
  # a2.set_yticks([])
  # a2.set_yticklabels([])
  # for spine in a1.spines.values():
  #     spine.set_visible(False)
  # for spine in a2.spines.values():
  #     spine.set_visible(False)
  #
  # #plt.savefig("wave_nnet_lfps.svg", transparent=True)
  #
  # plt.show()
  return


@app.cell(hide_code=True)
def _():
  from scipy.signal import butter, filtfilt, welch, coherence

  def compute_psd_lfp(lfp, fs, hp_cutoff=5.0, hp_order=3, nperseg=512):
    nyq = 0.5 * fs
    normal_cutoff = hp_cutoff / nyq
    b, a = butter(hp_order, normal_cutoff, btype='high', analog=False)
    lfp_hp = filtfilt(b, a, lfp)
    freqs, psd = welch(lfp_hp, fs=fs, nperseg=nperseg)

    return freqs, psd

  def compute_coh_lfp(x, y, fs, hp_cutoff=5.0, hp_order=3, nperseg=512):
    nyq = 0.5 * fs
    normal_cutoff = hp_cutoff / nyq
    b, a = butter(hp_order, normal_cutoff, btype='high', analog=False)
    x = filtfilt(b, a, x)
    y = filtfilt(b, a, y)
    freqs, coh = coherence(x, y, fs=fs, nperseg=nperseg)
    return freqs, coh

  # freqs, stn_psd = compute_psd_lfp(lfp_stn.sum(axis=1), 20000, nperseg=40000, hp_cutoff=10.)
  # stn_psd /= stn_psd.sum()
  # freqs, gpe_psd = compute_psd_lfp(lfp_gpe.sum(axis=1), 20000, nperseg=40000, hp_cutoff=10.)
  # gpe_psd /= gpe_psd.sum()
  # freqs, coh = compute_coh_lfp(lfp_stn.sum(axis=1), lfp_gpe.sum(axis=1), fs=20000, nperseg=40000, hp_cutoff=10.)
  #
  # _fig, (_a1, _a2) = plt.subplots(2, 1, figsize=(3.5, 6), tight_layout=True)
  # _a1.plot(freqs, gpe_psd, label='GPe', color="k", lw=1.8)
  # _a1.plot(freqs, stn_psd, label='STN', color=cmap(0.5), lw=1.8)
  # _a1.grid(True, which="both", alpha=0.7)
  # _a1.set_yscale('log')
  # _a1.set_xlim(0, 40)
  # _a1.set_ylim(1e-4, 1)  # 10^-4 bottom, 1 top
  # _a1.set_yticks([1e-4, 1e-2, 1])
  # _a1.tick_params(axis="both", labelsize=12)
  # _a1.legend(fontsize=12)
  # _a1.set_ylabel("Amplitude (a.u.)", fontsize=12)
  # _a1.set_xlabel("Frequency (Hz)", fontsize=12)
  # _a2.plot(freqs, coh, color=cmap(1.), lw=1.8)
  # _a2.set_xlim(0, 40)
  # _a2.tick_params(axis="both", labelsize=12)
  # _a2.set_ylabel("MS coherence", fontsize=12)
  # _a2.set_xlabel("Frequency (Hz)", fontsize=12)
  # _a2.set_yticks([0, 0.5, 1])
  # _a2.grid(True, which="both", alpha=0.7)
  # #plt.savefig("wave_nnet_coh.svg", transparent=True)
  # plt.show()
  return butter, coherence, compute_coh_lfp, compute_psd_lfp, filtfilt, welch


@app.cell(hide_code=True)
def _(butter, filtfilt, np):
  from scipy.signal import hilbert

  def bandpass(data, fs, low, high, order=4):
    nyq = fs / 2
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, data)

  def get_phase(data):
    return np.angle(hilbert(data))

  def get_envelope(data):
    return np.abs(hilbert(data))

  def detect_bursts(envelope, threshold_percentile=75):
    threshold = np.percentile(envelope, threshold_percentile)
    above = envelope > threshold
    edges = np.diff(above.astype(int))
    starts = np.where(edges == 1)[0]
    stops = np.where(edges == -1)[0]
    if stops.size == 0 or starts.size == 0:
      return []
    if stops[0] < starts[0]:
      stops = stops[1:]
    if starts[-1] > stops[-1]:
      starts = starts[:-1]
    return list(zip(starts, stops))

  def burst_locked_average(envelope, bursts, fs, win=(-0.3, 0.3), normalize=True):
    n_samples = int((win[1] - win[0]) * fs)
    locked = []
    for start, _ in bursts:
      center = start
      s = int(center + win[0] * fs)
      e = s + n_samples
      if s >= 0 and e <= len(envelope):
        segment = envelope[s:e]
        if normalize:
          max_val = np.max(segment)
          if max_val != 0:
            segment = segment / max_val
        locked.append(segment)
    return np.array(locked)

  def phase_diff_during_bursts(phase1, phase2, bursts):
    phase_diffs = []
    for start, stop in bursts:
      if stop <= len(phase1):
        diff = np.angle(np.exp(1j * (phase1[start:stop] - phase2[start:stop])))
        phase_diffs.extend(diff)
    return np.array(phase_diffs)

  return (
      bandpass,
      burst_locked_average,
      detect_bursts,
      get_envelope,
      get_phase,
      hilbert,
      phase_diff_during_bursts,
  )


@app.cell
def _(
    bandpass,
    burst_locked_average,
    detect_bursts,
    get_envelope,
    get_phase,
    np,
    phase_diff_during_bursts,
    spike_train_to_lfp,
):

  def lfp_avg(s):
    lfp = spike_train_to_lfp(s, 0.05, rayleigh_scale=50, kernel_duration=300)
    return lfp.sum(axis=1)

  def phase_thing(lfp_avg1, lfp_avg2):
    _fs = 2000
    low, high = 13, 30 # beta band

    lfp1 = bandpass(lfp_avg1, _fs, low, high)
    lfp2 = bandpass(lfp_avg2, _fs, low, high)

    env1 = get_envelope(lfp1)
    env2 = get_envelope(lfp2)
    phase1 = get_phase(lfp1)
    phase2 = get_phase(lfp2)

    # Detect bursts using one envelope (e.g., STN)
    bursts = detect_bursts(env1, threshold_percentile=75)

    # Compute burst-locked average envelope (for plotting like G/H)
    locked_env1 = burst_locked_average(env1, bursts, _fs)
    locked_env2 = burst_locked_average(env2, bursts, _fs)

    # Compute mean and SEM
    mean_env1 = np.mean(locked_env1, axis=0)
    mean_env2 = np.mean(locked_env2, axis=0)

    # Phase difference during bursts
    phase_diff = phase_diff_during_bursts(phase1, phase2, bursts)

    return mean_env1, mean_env2, phase_diff

  return lfp_avg, phase_thing


@app.cell
def _(lfp_avg, phase_thing, s_gpe1, s_stn1):
  lfp_stn_avg1 = lfp_avg(s_stn1)
  lfp_gpe_avg1 = lfp_avg(s_gpe1)
  mean_env_stn1, mean_env_gpe1, phase_diff1 = phase_thing(lfp_stn_avg1, lfp_gpe_avg1)
  return (
      lfp_gpe_avg1,
      lfp_stn_avg1,
      mean_env_gpe1,
      mean_env_stn1,
      phase_diff1,
  )


@app.cell
def _(lfp_avg, phase_thing, s_gpe2, s_stn2):
  lfp_stn_avg2 = lfp_avg(s_stn2)
  lfp_gpe_avg2 = lfp_avg(s_gpe2)
  mean_env_stn2, mean_env_gpe2, phase_diff2 = phase_thing(lfp_stn_avg2, lfp_gpe_avg2)
  return (
      lfp_gpe_avg2,
      lfp_stn_avg2,
      mean_env_gpe2,
      mean_env_stn2,
      phase_diff2,
  )


@app.cell
def _(
    mean_env_gpe1,
    mean_env_gpe2,
    mean_env_stn1,
    mean_env_stn2,
    plt,
    plt_phase_env,
):
  plt_phase_env(mean_env_stn1, mean_env_stn2)
  plt.savefig("env_stn.svg", transparent=True)
  plt.show()

  plt_phase_env(mean_env_gpe1, mean_env_gpe2)
  plt.savefig("env_gpe.svg", transparent=True)
  plt.show()
  return


@app.cell
def _(mean_env_stn2, np, plt):
  from scipy.ndimage import gaussian_filter1d

  def plt_phase_env(mean_env1, mean_env2, color="tab:blue", node="STN"):
    plt.figure(figsize=(3.5, 3.5), tight_layout=True)
    _t = np.linspace(-0.5, 0.5, mean_env_stn2.shape[0])
    plt.plot(_t, mean_env1, lw=3, color="#5e3c99")
    plt.plot(_t, mean_env2, lw=3, color="#3d9970")
    plt.xlabel("Time to burst onset (s)", fontsize=14)
    plt.ylabel("Normalised envelope (a.u.)", fontsize=14)

  def plt_phase_diff(phase_diff, color):
    fig = plt.figure(figsize=(4, 4), tight_layout=True)
    plt.axes(polar=True)
    bins = 100
    counts, bin_edges = np.histogram(phase_diff, bins=bins, range=(-np.pi, np.pi), density=True)
    counts = gaussian_filter1d(counts, sigma=2)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    bin_centers = np.append(bin_centers, bin_centers[0])
    counts = np.append(counts, counts[0])

    plt.plot(bin_centers, counts, color=color)
    plt.gca().set_yticklabels([])
    plt.tick_params(labelsize=12, pad=5)
    plt.grid(True, alpha=0.6)
    plt.hist(phase_diff, bins=100, density=True, alpha=0.4, color=color)
    plt.title("STN–GPe within-burst phase difference", fontsize=14)

  return gaussian_filter1d, plt_phase_diff, plt_phase_env


@app.cell
def _(plt):
  handles = [
      plt.Line2D([0], [0], color='#5e3c99', lw=8, label='Before Stimulation'),
      plt.Line2D([0], [0], color='#3d9970', lw=8, label='After Stimulation'),
  ]

  # Create a blank figure
  fig, ax = plt.subplots()
  ax.legend(handles=handles, loc='center', frameon=False, fontsize=16)

  # Hide axes
  ax.axis('off')
  plt.savefig("phase_legend.svg", transparent=True)
  plt.show()
  return ax, fig, handles


@app.cell
def _(phase_diff1, phase_diff2, plt, plt_phase_diff):
  plt_phase_diff(phase_diff2, color="#3d9970")
  plt.savefig("phase_after.svg", transparent=True)
  plt.show()

  plt_phase_diff(phase_diff1, color="#5e3c99")
  plt.savefig("phase_before.svg", transparent=True)
  plt.show()
  return


@app.cell
def _(
    cbgt,
    gpe_count,
    gpe_i_app,
    gpi_count,
    np,
    rand_gpe,
    rand_gpi,
    rand_stn,
    stn_count,
    time,
    w_n_net,
    white_noise_stn,
):
  np.random.seed(69)

  model2 = cbgt.Network(
      dt=0.05,
      total_t=8.,
      experiment="wave",
      gpe_i_app=gpe_i_app,
                                                  #   gpe_i_ext=gpe_i_ext,
                                                  #   stn_ca_pre=0,
                                                  #   stn_ca_post=0,
      stn_theta_d=0.01,
      stn_theta_p=0.05,
      stn_i_ext=white_noise_stn,
      stn_count=stn_count,
      gpe_count=gpe_count,
      gpi_count=gpi_count,
                                                  #   **w_cluster(stn_count, gpe_count, gpi_count),
      **w_n_net(stn_count, gpe_count, gpi_count),
      **rand_gpe(gpe_count),
      **rand_gpi(gpi_count),
      **rand_stn(stn_count),
  )
  _start = time()
  s_stn2, s_gpe2, s_gpi2 = model2.run_rk4()
  print(f"\n> Simulated 8s in {time() - _start:.2f}s 🚀\n")
  df2 = model2.to_polars()
  return df2, model2, s_gpe2, s_gpi2, s_stn2


if __name__ == "__main__":
  app.run()
