import cbgt

stn_i_ext = lambda t, n: -99.

if __name__ == "__main__":
  cbgt.RubinTerman.init_logger()
  rt = cbgt.RubinTerman(total_t=1.0,
                        stn_count=10,
                        gpe_count=10,
                        experiment="episodic",
                        stn_i_ext=stn_i_ext,
                        save_dir="experiments/test")
  df = rt.to_polars(2.)
  print(df["stn"].head())
  rt.run()
  rt.save_to_parquet_files("./experiments/test")
