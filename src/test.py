import cbgt
import numpy as np
from plot import *
from init import *
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import argparse
import time
import subprocess
import sys

np.random.seed(69)
target_color = "#5e3c99"
base = LinearSegmentedColormap.from_list("custom_purple", ["white", target_color], N=256)
cmap = ListedColormap(base(np.linspace(0.25, 1.0, 512)))
mpl.use('TkAgg')
cmap = "YlGnBu"


def random(rows, cols, n):
  res = np.zeros((rows, cols))

  for row in range(rows):
    ids = np.random.choice(cols, n, replace=False)
    res[row, ids] = 1

  return res


def nearest_no_loop(rows, cols, n):
  assert n % 2 == 1, "n must be odd for symmetric nearest neighbors"
  res = np.zeros((rows, cols), dtype=np.float64)

  ids = np.linspace(0, cols - 1, rows)
  ids = np.rint(ids).astype(int)

  half = n // 2
  for i, center in enumerate(ids):
    start = max(0, center - half)
    end = min(cols, center + half + 1)
    res[i, start:end] = 1.0

  return res


def nearest(rows, cols, n):
  assert n % 2 == 1
  diag = np.zeros((rows, cols)).astype(bool)
  res = np.zeros((rows, cols)).astype(bool)
  ids = np.linspace(0, cols - 1, rows)
  ids = np.rint(ids).astype(np.uint)
  diag[np.arange(rows), ids] = 1
  n += 1
  it = 0

  while n > 0:
    res |= np.roll(diag, it, axis=1)
    res |= np.roll(diag, -it, axis=1)
    it += 1
    n -= 2

  return res.astype(np.float64)


def parse_opt(pairs):
  kwargs = {}
  for pair in pairs:
    key, value = pair.split('=')
    kwargs[key] = eval(value)
  return kwargs


def zeros(time_ms, t):
  return 0.0


def ctx_stimuli(t, n):
  if (1000 < t < 1100 and n == 49):
    return 1.0
  return 0.0


def stn_i_ext(t, _):
  if (1000 < t < 1500):
    return 10.0
  return 0.0


def gpe_i_app(t, n):
  return 0.0


def test(experiment="wave", total_t=2, metrics=None, **opt):
  num_str = 10     # 50
  num_stn = 10     # 10
  num_gpe = 10     # 30
  num_gpi = 10     # 10
  num_ctx = 10     # 50

  num_str = 50
  num_stn = 10
  num_gpe = 30
  num_gpi = 10
  num_ctx = 50

  if metrics is None: metrics = []
  rt = cbgt.Network(
      dt=0.05,
      total_t=total_t,
      experiment=experiment,
      gpe_i_app=gpe_i_app,
      gpe_i_ext=zeros,
      gpi_i_ext=zeros,
      str_i_ext=zeros,
      stn_i_ext=stn_i_ext,
      ctx_stimuli=ctx_stimuli,
      stn_w_gpe=nearest_no_loop(num_gpe, num_stn, 1),
      stn_w_ctx=nearest_no_loop(num_ctx, num_stn, 3),
      str_w_str=nearest_no_loop(num_str, num_str, 9),
      str_w_ctx=nearest_no_loop(num_str, num_str, 1),
      gpe_w_g_g=nearest_no_loop(num_gpe, num_gpe, 15) - np.eye(num_gpe),
      gpe_w_s_g=nearest_no_loop(num_stn, num_gpe, 3),
      gpe_w_str=nearest_no_loop(num_str, num_gpe, 9),
                                                                         # gpi_w_g_g=random(num_gpi, num_gpi, 2),
      gpi_w_s_g=nearest_no_loop(num_stn, num_gpi, 3),
      gpi_w_str=nearest_no_loop(num_str, num_gpi, 9),
      str_count=num_str,
      stn_count=num_stn,
      gpe_count=num_gpe,
      gpi_count=num_gpi,
      ctx_count=num_ctx,
      **rand_str(num_str),
      **rand_stn(num_stn),
      **rand_gpe(num_gpe),
      **rand_gpi(num_gpi),
      **opt)
  start = time.time()
  rt.run_rk4()
  print(f"\n> Simulated {total_t:.2f}s in {time.time() - start:.2f}s 🚀\n")
  df = rt.to_polars()

  for y in metrics:
    if y == "i_ext":
      plot_time_activity([df for df in df.values()], [*df.keys()], y=y, cmap=f"YlGnBu_r")
      plt.show(block=False)
    elif y == "v":
      plot_time_activity([df for df in df.values()], [*df.keys()], y="v", unit="mV", vmin=-80, vmax=0, cmap=cmap)
      # plot_time_trace([df for df in df.values()], [*df.keys()], y="v", color="w")
      plt.show(block=False)
    else:
      plot_time_activity([df for df in df.values()], [*df.keys()], y=y, cmap=cmap)
      plt.show(block=False)
      #plot_time_trace([df for df in df.values()], [*df.keys()], y=y)
      #plt.show(block=False)

  plt.show()


def main(args):
  np.random.seed(69)
  opt = parse_opt(args.opt)
  plt.style.use('dark_background')
  plt.rcParams["figure.facecolor"] = "#101418"
  plt.rcParams["axes.facecolor"] = "#101418"
  mpl.rcParams['savefig.transparent'] = True
  #mpl.rcParams['figure.facecolor'] = 'none'
  #mpl.rcParams['axes.facecolor'] = 'none'
  test(experiment=args.experiment, total_t=args.time, metrics=args.plt, **opt)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('experiment', type=str)
  parser.add_argument('time', type=float)
  parser.add_argument('--opt', nargs='*', default=[], help='Key=value pairs')
  parser.add_argument('--plt', nargs='*', default=[], help='Key=value pairs')
  parser.add_argument('--subprocess', action=argparse.BooleanOptionalAction)
  args = parser.parse_args()
  if args.subprocess:
    main(args)
  else:
    subprocess.Popen([sys.executable, __file__] + sys.argv[1:] + ["--subprocess"])
