import marimo

__generated_with = "0.11.20"
app = marimo.App(
    width="columns",
    app_title="Experiment",
    layout_file="layouts/experiment.slides.json",
    css_file="/home/aaron/.config/marimo/custom.css",
)


@app.cell(column=0, hide_code=True)
def _(mo):
    mo.md(r"""# Cortical Basal Ganglia model demo""")
    return


@app.cell(hide_code=True)
def _(cmap, df_episodic, plot_time_activity):
    plot_time_activity([df_episodic["stn"]["v"], df_episodic["gpe"]["v"]], ["STN", "GPe"],
                       2.0, title="Episodic", cmap=cmap)
    return


@app.cell(hide_code=True)
def _(df_episodic, plot_voltage_trace):
    plot_voltage_trace([df_episodic["stn"]["v"], df_episodic["gpe"]["v"]], ["STN", "GPe"], 2.0, title="Episodic Trace", color=(0.435, 0.886, 0.973, 0.7))
    return


@app.cell
def _(cbgt, np):
    def episodic():
        np.random.seed(69)

        gpe_count, stn_count = 10, 10
        gpe_i_app = lambda t, n: -1.2

        c_g_s = np.zeros((gpe_count, stn_count), dtype=np.float64)
        c_s_g = np.zeros((stn_count, gpe_count), dtype=np.float64)
        c_g_g = (~np.eye(gpe_count, dtype=np.bool)).astype(np.float64)

        for idx_g in range(gpe_count):
            ids_s = np.random.choice(np.arange(stn_count), size=3, replace=False)
            c_g_s[idx_g, ids_s] = 1

        c_s_g[np.arange(stn_count), np.random.permutation(gpe_count)] = 1

        rt = cbgt.RubinTerman(dt=0.01, total_t=2, stn_count=stn_count, gpe_count=gpe_count,
                              gpe_i_app=gpe_i_app, experiment="episodic",
                              stn_c_g_s=c_g_s, gpe_c_s_g=c_s_g, gpe_c_g_g=c_g_g,
                              # gpe_g_g_g=0.06, gpe_g_s_g=0.03, stn_g_g_s=2.5
        )
        return rt.run(2.0), rt

    df_episodic, rt_episodic = episodic()
    return df_episodic, episodic, rt_episodic


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import cbgt
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np
    from src.plot import plot_time_activity, plot_voltage_trace

    plt.rcParams['figure.facecolor'] = 'none'
    plt.rcParams['axes.facecolor'] = 'none'

    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "transparent_blue",
        [(0, (0.435, 0.886, 0.973, 0)),
         (1, (0.435, 0.886, 0.973, 1))]
    )

    cbgt.RubinTerman.init_logger("warn")

    mo.md(f"""
    ###Package versions</br>

    - **marimo** {mo.__version__}
    - **matplotlib** {mpl.__version__}
    - **numpy** {np.__version__}
    """)
    return cbgt, cmap, mo, mpl, np, plot_time_activity, plot_voltage_trace, plt


@app.cell(column=1)
def _():
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
        rt_cluster = cbgt.RubinTerman(dt=0.01, total_t=2, experiment=radio.value)
        df_cluster =  rt_cluster.run(1.0)
        plot_time_activity([df_cluster["stn"]["v"], df_cluster["gpe"]["v"]], ["STN", "GPe"], 
                           1.0, title=f"Continous {radio.value}", cmap=cmap)
    return df_cluster, rt_cluster


if __name__ == "__main__":
    app.run()
