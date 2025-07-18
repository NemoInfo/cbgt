import numpy as np
from scipy.stats import rayleigh
import matplotlib.pyplot as plt

# --- Configuration Parameters ---
num_time_steps = 1000
num_neurons = 5
spike_probability = 0.005
rayleigh_scale = 10
kernel_length = 100

# Generate a synthetic spike train (T x N)
spike_train = (np.random.rand(num_time_steps, num_neurons) < spike_probability).astype(int)

# Create the Rayleigh kernel
x = np.arange(0, kernel_length)
kernel = rayleigh.pdf(x, scale=rayleigh_scale)
kernel = kernel / np.sum(kernel) # Normalize kernel

# Initialize array for LFP signals
lfp_signals = np.zeros_like(spike_train, dtype=float)

# Convolve each neuron's spike train with the kernel
for i in range(num_neurons):
  lfp_signals[:, i] = np.convolve(spike_train[:, i], kernel, mode='same')

plt.plot(lfp_signals[:, 0])
plt.show()
