import cbgt
import numpy as np
from sys import argv
from plot import *
import polars as pl
import matplotlib.pyplot as plt
import matplotlib as mpl


def test2():
  np.random.seed(69)

  rt = cbgt.Network(dt=0.01, total_t=2, experiment="cluster")
  rt.run_euler()
  df = rt.to_polars()

  df_ca_gpe = df["gpe"].with_columns(pl.col("ca_s_g").arr.get(2).alias("ca0"))
  df_ca_stn = df["stn"].with_columns(pl.col("ca_g_s").arr.get(2).alias("ca0"))

  plot_time_activity([df_ca_stn, df_ca_gpe], ["stn", "gpe"], y="ca0", title="Synaptic Ca", vmin=0, vmax=None)
  plot_time_activity([df["stn"]], ["stn"], y="ca", title="Soma Ca", vmin=0, vmax=None)
  plot_time_activity([df["gpe"]], ["gpe"], y="ca", title="Soma Ca", vmin=0, vmax=None)

  plt.show()


if __name__ == "__main__":
  plt.rcParams["figure.facecolor"] = "none"
  plt.rcParams["axes.facecolor"] = "none"
  cbgt.RubinTerman.init_logger("warn")
  if len(argv) == 2:
    if argv[1] == "1":
      pass
    elif argv[1] == "2":
      test2()
    else:
      exit(1)
  else:
    exit(1)
