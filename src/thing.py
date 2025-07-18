from plot import plot_time_activity
import polars as pl
import matplotlib.pyplot as plt

stn = pl.read_parquet("test/stn.parquet")
gpe = pl.read_parquet("test/gpe.parquet")
df = {"stn": stn, "gpe": gpe}

plot_time_activity(list(df.values()), list(df.keys()), y="v", max_num_neurons=100)
plot_time_activity(list(df.values()), list(df.keys()), y="ca", vmin=0, max_num_neurons=100)
plt.show()
