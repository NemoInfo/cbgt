import cbgt
import numpy as np

gpe_i_app = lambda t, n: -1.2

if __name__ == "__main__":
  cbgt.RubinTerman.init_logger()
  rt = cbgt.RubinTerman(total_t=2.0, experiment="cluster", gpe_i_app=gpe_i_app)

  rt.run()
  df = rt.to_polars(2.)
  print(df["gpe"])
  print(np.array(df["gpe"]["w_g_g"][0]).reshape((8, 8)).__repr__())
