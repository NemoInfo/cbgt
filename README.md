## CBGT Master's project

## How to use
Starting in the root directory, set up the environment in the project directory:

#### Virtual environment

Create a python virtual environment (if you want to use nix make sure to name it "venv"):
```sh
virtualenv venv
```

I use [Nix Flakes](https://nixos.wiki/wiki/Flakes) for dependencies. (this will also source the venv)
```sh
nix develop .
```

### To build the python bindings
```sh
maturin develop --release
```

#### Python example
```python3
rt = cbgt.RubinTerman(dt=0.01, total_t=2)
data = rt.run()
print(print(data["stn"]["v"]).shape)
```
Experiment graphs will be kept in [the Jupyter Notebook](/experiment.ipynb). You can refer to it for more exampels.

### To build & run rust backend bindings
```sh
cargo run --relese
```

#### Rust example
```rust
let res = RubinTerman {
  dt: 0.01,
  total_t: 2.,
  ..Default::default()
}._run();
```
