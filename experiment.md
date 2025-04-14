---
title: Experiment
marimo-version: 0.11.20
width: medium
css_file: /home/aaron/.config/marimo/custom.css
---

# Cortical Basal Ganglia model demo

```python {.marimo}
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
                          gpe_g_g_g=0.06, gpe_g_s_g=0.03, stn_g_g_s=2.5
    )
    return rt.run(), rt

df_episodic, rt_episodic = episodic()
```

```python {.marimo}
plot_time_activity([df_episodic["stn"]["v"], df_episodic["gpe"]["v"]], ["STN", "GPe"],
                   rt_episodic.dt, title="Episodic", cmap=cmap)
```

```python {.marimo}
def wave():
    gpe_count, stn_count = 10, 10
    gpe_i_app = lambda t, n: -1.2

    c_g_s = np.zeros((gpe_count, stn_count), dtype=np.float64)
    c_s_g = np.zeros((stn_count, gpe_count), dtype=np.float64)
    c_g_g = (~np.eye(gpe_count, dtype=np.bool)).astype(np.float64)

    c_s_g[0, [0,1,-1]] = 1
    for i in range(stn_count-1):
        c_s_g[i+1] = np.roll(c_s_g[i], 1)

    c_g_s[0, [0,1,2,-1,-2]] = 1
    for i in range(gpe_count-1):
        c_g_s[i+1] = np.roll(c_g_s[i], 1) 

    rt = cbgt.RubinTerman(dt=0.01, total_t=2, stn_count=stn_count, gpe_count=gpe_count,
                          gpe_i_app=gpe_i_app, experiment="wave_rt",
                          stn_c_g_s=c_g_s, gpe_c_s_g=c_s_g, gpe_c_g_g=c_g_g)
    return rt.run(), rt

df_wave, rt_wave = wave()
```

```python {.marimo}
plot_time_activity([df_wave["stn"]["v"], df_wave["gpe"]["v"]], ["STN", "GPe"], 
                   rt_wave.dt, title="Continous wave", cmap=cmap)
```

```python {.marimo}
def cluster():
    gpe_count, stn_count = 8, 8
    gpe_i_app = lambda t, n: -.6

    c_g_s = np.zeros((gpe_count, stn_count), dtype=np.float64)
    c_s_g = np.eye((stn_count), dtype=np.float64)
    c_g_g = np.zeros((gpe_count, gpe_count), dtype=np.float64)

    c_g_g[0, [1,-1]] = 1
    for i in range(stn_count-1):
        c_g_g[i+1] = np.roll(c_g_g[i], 1)

    c_g_s[0, [2,-2]] = 1
    for i in range(stn_count-1):
        c_g_s[i+1] = np.roll(c_g_s[i], 1)

    rt = cbgt.RubinTerman(dt=0.01, total_t=2, gpe_i_app=gpe_i_app, stn_count=stn_count, gpe_count=gpe_count,
                          experiment="cluster",
                          stn_c_g_s=c_g_s, gpe_c_s_g=c_s_g, gpe_c_g_g=c_g_g)
    return rt.run(), rt

df_cluster, rt_cluster = cluster()
```

```python {.marimo}
plot_time_activity([df_cluster["stn"]["v"], df_cluster["gpe"]["v"]], ["STN", "GPe"], 
                   rt_cluster.dt, title="Continous cluster", cmap=cmap)
```

```python {.marimo}
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

cbgt.RubinTerman.init_logger("error")

mo.md(f"""
###Package versions</br>

- **marimo** {mo.__version__}
- **matplotlib** {mpl.__version__}
- **numpy** {np.__version__}
""")
```