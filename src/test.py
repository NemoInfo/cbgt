import cbgt
import numpy as np
from sys import argv
from plot import plot_time_activity

ITERATIONS = 1000


def episodic():
  np.random.seed(69)

  gpe_count, stn_count = 10, 10
  rt = cbgt.RubinTerman(dt=0.01, total_t=0.01 * 1e-3 * (ITERATIONS + 1), experiment="episodic")
  rt.run()


def episodic2():
  np.random.seed(69)

  gpe_count, stn_count = 10, 10
  rt = cbgt.Network(dt=0.05, total_t=2, experiment="cluster")
  rt.run_rk4()
  #rt.run_euler()
  df = rt.to_polars()
  plot_time_activity(
      [df["stn"], df["gpe"]],
      ["STN", "GPe"],
      title="Episodic",
      y="ca",
      vmin=0,
      vmax=None,
  )


if __name__ == "__main__":
  cbgt.RubinTerman.init_logger("warn")
  if len(argv) == 2:
    if argv[1] == "1":
      episodic()
    elif argv[1] == "2":
      episodic2()
    else:
      exit(1)
  else:
    exit(1)
