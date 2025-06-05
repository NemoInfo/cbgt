import cbgt
import numpy as np
from sys import argv
from plot import *
import polars as pl
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse
import time
import subprocess
import sys

cmap = "YlGnBu"


def parse_opt(pairs):
  kwargs = {}
  for pair in pairs:
    key, value = pair.split('=')
    kwargs[key] = float(value) if value.isnumeric() else value
  return kwargs


def gpi_i_ext(time_ms, n):
  return 0.0


def gpe_i_ext(time_ms, n):
  start, end = 500, 2500
  if time_ms < start or time_ms > end:
    return 0.0

  period_ms = 1000.0 / 133.0
  time_in_cycle = (time_ms - start) % period_ms

  if time_in_cycle < (period_ms / 2):
    return -20.0
  else:
    return 0.0


def stn_i_ext(time_ms, n):
  # start, end = 1500, 2500
  # if time_ms < start or time_ms > end:
  #   return 0.0
  #
  # period_ms = 1000.0 / 133.0
  # time_in_cycle = (time_ms - start) % period_ms
  #
  # if time_in_cycle < (period_ms / 2):
  #   return 30.0
  # else:
  #   return 0.0
  return 0.0


def test(experiment="wave", total_t=2, metrics=None, **opt):
  if metrics is None: metrics = []
  rt = cbgt.Network(dt=0.05,
                    total_t=total_t,
                    experiment=experiment,
                    gpe_i_ext=gpe_i_ext,
                    gpi_i_ext=gpi_i_ext,
                    stn_i_ext=gpe_i_ext,
                    gpi_w_g_g=np.eye(8, dtype=np.float64),
                    **opt)
  start = time.time()
  rt.run_rk4()
  print(f"{time.time() - start:.2f}s")
  df = rt.to_polars()

  # df_ca_gpe = df["gpe"].with_columns(pl.col("ca_s_g").arr.get(2).alias("ca0"))
  # df_ca_stn = df["stn"].with_columns(pl.col("ca_g_s").arr.get(2).alias("ca0"))
  # plot_time_activity([df_ca_stn, df_ca_gpe], ["stn", "gpe"], y="ca0", title="Synaptic Ca", cmap=cmap)

  for y in metrics:
    if y == "i_ext":
      plot_time_activity([df for df in df.values()], [*df.keys()], y=y, cmap=f"YlGnBu_r")
      plt.show(block=False)
    elif y == "v":
      plot_time_activity([df for df in df.values()], [*df.keys()], y="v", unit="mV", vmin=-80, vmax=0, cmap=cmap)
      plt.show(block=False)
    else:
      plot_time_activity([df for df in df.values()], [*df.keys()], y=y, cmap=cmap)
      plt.show(block=False)

  plt.show()


def main(args):
  np.random.seed(69)
  opt = parse_opt(args.opt)
  plt.style.use('dark_background')
  plt.rcParams["figure.facecolor"] = "#101418"
  plt.rcParams["axes.facecolor"] = "#101418"
  test(experiment=args.experiment, total_t=args.time, metrics=args.plt, **opt)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('experiment', type=str)
  parser.add_argument('time', type=int)
  parser.add_argument('--opt', nargs='*', default=[], help='Key=value pairs')
  parser.add_argument('--plt', nargs='*', default=[], help='Key=value pairs')
  parser.add_argument('--subprocess', action=argparse.BooleanOptionalAction)
  args = parser.parse_args()
  if args.subprocess:
    main(args)
  else:
    subprocess.Popen([sys.executable, __file__] + sys.argv[1:] + ["--subprocess"])
