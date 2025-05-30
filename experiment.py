import marimo

__generated_with = "0.11.20"
app = marimo.App(
    width="columns",
    app_title="Experiment",
    css_file="/home/aaron/.config/marimo/custom.css",
)


@app.cell(column=0, hide_code=True)
def _(mo):
    mo.md(r"""# Cortical Basal Ganglia model demo""")
    return


@app.cell(hide_code=True)
def _(df_episodic, plot_time_trace):
    plot_time_trace(
        [df_episodic["stn"], df_episodic["gpe"]],
        ["STN", "GPe"],
        title="Episodic Trace",
        color=(0.435, 0.886, 0.973, 0.7),
    )
    return


@app.cell
def _(cmap, df_episodic, plot_time_activity):
    plot_time_activity(
        [df_episodic["stn"]],
        ["STN"],
        title="Episodic",
        cmap=cmap,
        y="ca", vmin=0, vmax=None
    )
    plot_time_activity(
        [df_episodic["gpe"]],
        ["GPe"],
        title="Episodic",
        cmap=cmap,
        y="ca", vmin=0, vmax=None
    )
    return


@app.cell(hide_code=True)
def _(cmap, df_episodic, plot_time_activity):
    plot_time_activity(
        [df_episodic["stn"], df_episodic["gpe"]],
        ["STN", "GPe"],
        title="Episodic",
        cmap=cmap,
    )
    return


@app.cell
def _(cbgt, np):
    def episodic():
        np.random.seed(69)

        gpe_count, stn_count = 10, 10
        rt = cbgt.RubinTerman(dt=0.01, total_t=2, experiment="cluster")
        rt.run()
        return rt.to_polars(), rt


    df_episodic, rt_episodic = episodic()
    return df_episodic, episodic, rt_episodic


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import cbgt
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np
    from src.plot import plot_time_activity, plot_time_trace

    plt.rcParams["figure.facecolor"] = "none"
    plt.rcParams["axes.facecolor"] = "none"
    mpl.rcParams["figure.dpi"] = 300

    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "transparent_blue",
        [(0, (0.435, 0.886, 0.973, 0)), (1, (0.435, 0.886, 0.973, 1))],
    )
    cmap = "YlGnBu"
    cbgt.RubinTerman.init_logger("error")

    mo.md(f"""
    ###Package versions</br>

    - **marimo** {mo.__version__}
    - **matplotlib** {mpl.__version__}
    - **numpy** {np.__version__}
    """)
    return cbgt, cmap, mo, mpl, np, plot_time_activity, plot_time_trace, plt


@app.cell(column=1)
def _(cbgt, np):
    def episodic2():
        np.random.seed(69)

        gpe_count, stn_count = 10, 10
        rt = cbgt.Network(dt=0.01, total_t=2, experiment="episodic")
        rt.run_euler()
        return rt.to_polars(), rt


    df_episodic2, rt_episodic2 = episodic2()
    return df_episodic2, episodic2, rt_episodic2


@app.cell
def _(cmap, df_episodic2, plot_time_activity):
    plot_time_activity(
        [df_episodic2["stn"], df_episodic2["gpe"]],
        ["STN", "GPe"],
        title="Episodic",
        cmap=cmap,
        y="v",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    options = ["wave", "cluster"]
    radio = mo.ui.radio(options=options, value="wave", label="## Experiment")

    radio
    return options, radio


@app.cell(hide_code=True)
def _(cbgt, cmap, plot_time_activity, radio):
    if radio.value:
        rt_cluster = cbgt.RubinTerman(
            dt=0.01,
            total_t=5,
            experiment=radio.value,
            # gpe_a_pre=0.0, gpe_a_post=0., stn_a_pre=0, stn_a_post=0
        )
        rt_cluster.run()
        df_cluster = rt_cluster.to_polars(2.0)
        plot_time_activity(
            [df_cluster["stn"], df_cluster["gpe"]],
            ["STN", "GPe"],
            title=f"Continous {radio.value}",
            cmap=cmap,
        )
    return df_cluster, rt_cluster


if __name__ == "__main__":
    app.run()
