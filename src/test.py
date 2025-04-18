import cbgt

print(cbgt.__file__)

if __name__ == "__main__":
  cbgt.RubinTerman.init_logger()
  rt = cbgt.RubinTerman(total_t=1.0, experiment="cluster")
  df = rt.to_polars(2.)
  print(df["stn"].head())
  rt.run()
  rt.save_to_parquet_files("./test/")
