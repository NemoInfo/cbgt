import cbgt

gpe_i_app = lambda t, n: -1.4


def episodic():
  cbgt.RubinTerman.init_logger()

  rt = cbgt.RubinTerman(dt=0.01, total_t=2, experiment="episodic", gpe_i_ext=gpe_i_app)
  rt.run()
  return rt.to_polars(2.), rt


if __name__ == "__main__":
  df_episodic, rt_episodic = episodic()
