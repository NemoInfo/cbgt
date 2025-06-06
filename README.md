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
If you don't want to use Nix Flakes, check the required pakages in [flake.nix](/flake.nix), and install them manually.

Then, to build the python bindings
```sh
maturin develop --release
```

#### Python example
```python3
rt = cbgt.RubinTerman(dt=0.01, total_t=2)
data = rt.run()
print((data["stn"]["v"]).shape)
```
Experiment graphs will be kept in [the Marimo Notebook](/experiment.py). You can refer to it for more exampels.

Make sure to start mairmo like this (from the venv) so the cbgt module is available. 
```sh
python3 -m marimo edit
```

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
