import cbgt
import numpy as np
from plot import *
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import argparse
import time
import subprocess
import sys

target_color = "#5e3c99"
base = LinearSegmentedColormap.from_list("custom_purple", ["white", target_color], N=256)
cmap = ListedColormap(base(np.linspace(0.25, 1.0, 512)))
mpl.use('TkAgg')


def parse_opt(pairs):
  kwargs = {}
  for pair in pairs:
    key, value = pair.split('=')
    kwargs[key] = eval(value)
  return kwargs


def zeros(time_ms, t):
  return 0.0


def uniform(time_ms, n):
  start, end = 500, 2500
  return 40. if start < time_ms < end else 0


def gpi_i_ext(time_ms, n):
  return 0.0


def oscillatory(time_ms, n):
  start, end = 500, 2500
  if time_ms < start or time_ms > end:
    return 0.0

  period_ms = 1000.0 / 44.0
  time_in_cycle = (time_ms - start) % period_ms

  if time_in_cycle < (period_ms / 2):
    return 50.0
  else:
    return 0.0


def white_noise_stn(time_ms, n):
  import numpy as np
  return np.random.normal(loc=10, scale=4)


def white_noise_str(time_ms, n):
  import numpy as np
  # return np.random.normal(loc=-8, scale=4)
  return 0


def white_noise_gpe(time_ms, n):
  import numpy as np
  return np.random.normal(loc=0, scale=0.1)


def brown_noise_str(t, n):
  from noise import pnoise2
  return abs(pnoise2(t * 0.01, n, octaves=8, persistence=0.5, lacunarity=2.0)) * 20


def brown_noise_stn(t, n):
  from noise import pnoise2
  return abs(pnoise2(t * 0.01, n, octaves=8, persistence=0.5, lacunarity=2.0)) * 10


def test(experiment="wave", total_t=2, metrics=None, **opt):
  if metrics is None: metrics = []
  rt = cbgt.Network(dt=0.05,
                    total_t=total_t,
                    experiment=experiment,
                    stn_i_ext=brown_noise_stn,
                    str_i_ext=brown_noise_str,
                    gpi_i_ext=zeros,
                    **opt)
  start = time.time()
  rt.run_rk4()
  print(f"\n> Simulated {total_t:.2f}s in {time.time() - start:.2f}s 🚀\n")
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
  # plt.style.use('dark_background')
  # plt.rcParams["figure.facecolor"] = "#101418"
  # plt.rcParams["axes.facecolor"] = "#101418"
  mpl.rcParams['savefig.transparent'] = True
  mpl.rcParams['figure.facecolor'] = 'none'
  mpl.rcParams['axes.facecolor'] = 'none'
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
