import marimo

__generated_with = "0.11.20"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import cbgt
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from src.plot import plot_time_activity, plot_time_trace
    from time import time
    return cbgt, mo, mpl, np, plot_time_activity, plot_time_trace, plt, time


@app.cell
def _(cbgt):
    model = cbgt.Network(dt=0.05, total_t=4., experiment="cluster",
                         gpe_g_g_g = 0.0002,
                         gpe_g_s_g = 0.0001,
                        )
    model.run_rk4()
    df = model.to_polars()
    return df, model


@app.cell
def _(mpl, np):
    target_color = "#5e3c99"
    base = mpl.colors.LinearSegmentedColormap.from_list("custom_purple", ["white", target_color], N=256)
    cmap = mpl.colors.ListedColormap(base(np.linspace(0.25, 1.0, 512)))
    return base, cmap, target_color


@app.cell
def _(cmap, df, plot_time_activity, plt):
    plot_time_activity([*df.values()], ["STN", "GPe", "GPi"], y="ca", cmap=cmap, vmax=10)
    _fig = plt.gcf()
    _fig.set_size_inches(6.5, 5)
    plt.savefig("cluster_intro.svg", transparent=True)
    plt.show()
    return


if __name__ == "__main__":
    app.run()
