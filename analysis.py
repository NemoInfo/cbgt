import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch

gpe = pl.read_parquet("test_out/gpe.parquet")

# elementwise average of three list[f64] columns i_stn, i_str, i_gpe -> new column i_avg
gpe = gpe.with_columns(
    ((pl.col("i_stn").list.mean() + pl.col("i_str").list.mean() + pl.col("i_gpe").list.mean()) / 3).alias("i_avg"))

g = gpe.slice(5000)

# --- IQR clipping ---
q1 = g.select(pl.col("i_avg").quantile(0.05)).to_series()[0]
q3 = g.select(pl.col("i_avg").quantile(0.95)).to_series()[0]
iqr = q3 - q1
lower_iqr = q1 - 1.5 * iqr
upper_iqr = q3 + 1.5 * iqr

g = g.with_columns(pl.col("i_avg").clip(lower_iqr, upper_iqr).alias("i_avg_clipped_iqr"))
x_clipped = g["i_avg_clipped_iqr"]

# bandpass 4-100 Hz (Butterworth, zero-phase)
fs = 1000.0
lowcut, highcut = 4.0, 100.0
order = 4
b, a = butter(order, [lowcut / (fs / 2), highcut / (fs / 2)], btype='band')
x_bp = filtfilt(b, a, x_clipped)

# PSD (Welch)
nperseg = 4096 if len(x_bp) >= 4096 else 1024
f, Pxx = welch(x_bp, fs=fs, nperseg=nperseg, detrend='constant', scaling='density')

# restrict to 4-100 Hz for plotting
mask = (f >= lowcut) & (f <= highcut)
f_plot, Pxx_plot = f[mask], Pxx[mask]

plt.figure(figsize=(8, 5))
plt.subplot(2, 1, 1)
t = np.arange(len(x_bp)) / fs
plt.plot(t, x_clipped, label='clipped', alpha=0.5)
plt.plot(t, x_bp, label='bandpass 4-100 Hz', linewidth=1)
plt.legend()
plt.xlabel('Time (s)')

plt.subplot(2, 1, 2)
plt.semilogy(f_plot, Pxx_plot)
plt.xlim(0, 60)
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD')
plt.title('GPe Welch PSD (4–100 Hz)')
plt.grid(True)
plt.tight_layout()
plt.show(block=False)

gpe = pl.read_parquet("test_out/stn.parquet")

# elementwise average of three list[f64] columns i_stn, i_str, i_gpe -> new column i_avg
gpe = gpe.with_columns(((pl.col("i_gpe").list.mean() + pl.col("i_ctx").list.mean()) / 3).alias("i_avg"))

g = gpe.slice(5000)

# --- IQR clipping ---
q1 = g.select(pl.col("i_avg").quantile(0.05)).to_series()[0]
q3 = g.select(pl.col("i_avg").quantile(0.95)).to_series()[0]
iqr = q3 - q1
lower_iqr = q1 - 1.5 * iqr
upper_iqr = q3 + 1.5 * iqr

g = g.with_columns(pl.col("i_avg").clip(lower_iqr, upper_iqr).alias("i_avg_clipped_iqr"))
x_clipped = g["i_avg_clipped_iqr"]

# bandpass 4-100 Hz (Butterworth, zero-phase)
fs = 1000.0
lowcut, highcut = 4.0, 100.0
order = 4
b, a = butter(order, [lowcut / (fs / 2), highcut / (fs / 2)], btype='band')
x_bp = filtfilt(b, a, x_clipped)

# PSD (Welch)
nperseg = 4096 if len(x_bp) >= 4096 else 1024
f, Pxx = welch(x_bp, fs=fs, nperseg=nperseg, detrend='constant', scaling='density')

# restrict to 4-100 Hz for plotting
mask = (f >= lowcut) & (f <= highcut)
f_plot, Pxx_plot = f[mask], Pxx[mask]

plt.figure(figsize=(8, 5))
plt.subplot(2, 1, 1)
t = np.arange(len(x_bp)) / fs
plt.plot(t, x_clipped, label='clipped', alpha=0.5)
plt.plot(t, x_bp, label='bandpass 4-100 Hz', linewidth=1)
plt.legend()
plt.xlabel('Time (s)')

plt.subplot(2, 1, 2)
plt.semilogy(f_plot, Pxx_plot)
plt.xlim(0, 60)
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD')
plt.title('STN Welch PSD (4–100 Hz)')
plt.grid(True)
plt.tight_layout()
plt.show()
