import marimo

__generated_with = "0.11.20"
app = marimo.App(width="medium")


@app.cell
def _():
    import polars as pl
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    return mpl, np, pl, plt


@app.cell
def _(pl):
    stn = pl.read_parquet("test_out/stn.parquet")
    stn = stn.filter(pl.col("time") > 5300)
    return (stn,)


@app.cell
def _(np, stn, voltage_to_spike):
    v = np.array(stn["v"].to_numpy().tolist())
    spikes = np.vectorize(voltage_to_spike)(v)
    spikes.sum()
    return spikes, v


@app.cell
def _(min_si, np):
    def voltage_to_spike(v, theta=-20):
        if v > theta: return 1
        return 0

    def isi(spikes, min_isi=2, dt=0.05):
        new_spikes = np.zeros_like(spikes)
        li = None
        for i in enumerate(spikes):
            if i > li + min_si * dt:
                li = i
                spikes[i] = 1

        return spikes
    return isi, voltage_to_spike


@app.cell
def _(plt, stn, v):
    plt.plot(stn["time"]-5500, v[:, 9])
    return


@app.cell
def _(plt, spikes, stn):

    _fig, _ax = plt.subplots(1, 1, figsize=(8 / 2.54, 4 / 2.54), sharex=True)
    for _i, _spike in enumerate(spikes.T[::-1]): 
        t_spikes = stn["time"].to_numpy()[_spike == 1]
        _ax.plot(t_spikes, _spike[_spike == 1] * _i , ".", color="white", markersize=2)

    plt.show()
    return (t_spikes,)


if __name__ == "__main__":
    app.run()
