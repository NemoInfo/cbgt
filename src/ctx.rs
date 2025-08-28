// STIMULI (T, N) -> RATE_STIMULI (T, N)
// UNIFORM (T, N) & RATE_STIMULI -> SPIKES
// SPIKES -> SPIKES_ENFORCED_MIN_ISI
// SPIKES * RAYLEIGH -> SYNAPTIC INPUT

// SYNAPTIC INPUT * CONNECTION MATRIX * REVERSAL POTENTIAL -> CURRENT

use ndarray::{Array1, Array2, ArrayView1};
use ndarray_rand::{rand::SeedableRng, rand_distr::Open01, RandomExt};
use pyo3::{PyAny, Python};
use struct_field_names_as_array::FieldNamesAsSlice;

use crate::{
  types::{Boundary, Build, Builder, Neuron, NeuronConfig},
  util::{array2_to_polars_column, toml_py_function_qualname_to_py_object, vectorize_i_ext_py},
  CTXParameters,
};

#[derive(Default, Debug)]
pub struct CTX;

impl Neuron for CTX {
  const TYPE: &'static str = "CTX";
}

#[derive(FieldNamesAsSlice, Debug)]
pub struct CTXPopulationBoundryConditions {
  pub count: usize,
  pub stimuli: pyo3::Py<PyAny>,
}

pub struct CTXHistory {
  pub s: Array2<f64>,
  pub stimuli_f: pyo3::Py<PyAny>,
}

impl CTXHistory {
  pub fn new(
    num_timesteps: usize,
    ctx_count: usize,
    edge_resolution: usize,
    bc: CTXPopulationBoundryConditions,
  ) -> Self {
    Self { s: Array2::zeros((num_timesteps * edge_resolution, ctx_count)), stimuli_f: bc.stimuli }
  }

  pub fn fill_s(&mut self, py: Python, p: &CTXParameters, start_time: f64, end_time: f64, dt: f64) {
    let stim = vectorize_i_ext_py(py, &self.stimuli_f, start_time, end_time, dt, self.s.shape()[1]);
    let rates = stimuli_to_firing_rates(&stim, p.stimulated_rate, p.base_rate, p.sig_s, p.max_rate);
    let spikes = firing_rates_to_cortical_spikes(&rates, dt);
    let spikes = enforce_min_inter_spike_interval(&spikes, dt);
    let syn_ctx = convolve_spike_train(&spikes, p.syn_rayleigh_sig, dt, p.syn_kernel_len);
    let max_val = syn_ctx.fold(f64::MIN, |a, &b| a.max(b));
    self.s = syn_ctx / max_val;
  }

  pub fn into_compressed_polars_df(
    &self,
    idt: f64,
    odt: Option<f64>,
    edge_resolution: usize,
    data: &Vec<&str>,
    start_time: f64,
  ) -> polars::prelude::DataFrame {
    let num_timesteps = self.s.nrows() / edge_resolution;
    let odt = odt.unwrap_or(1.); // ms
    let step = odt / idt;
    if step != step.trunc() {
      log::warn!(
        "output_dt / simulation_dt = {step} is not integer. With a step of {} => output_dt = {}",
        step.trunc(),
        step.trunc() * idt
      );
    }
    let step = step.trunc() as usize;
    let output_dt = step as f64 * idt;
    let erange = ndarray::s![0..num_timesteps * edge_resolution;step * edge_resolution, ..];
    let num_timesteps = (0..num_timesteps).step_by(step).len();
    let time = start_time
      + (Array1::range(0., num_timesteps as f64, 1.) * output_dt).to_shape((num_timesteps, 1)).unwrap().to_owned();

    let mut out = vec![];
    if data.contains(&"time") {
      out.push(array2_to_polars_column("time", time.view()))
    }
    if data.contains(&"s") {
      out.push(array2_to_polars_column("s", self.s.slice(erange)))
    }

    polars::prelude::DataFrame::new(out).expect("This shouldn't happend if the struct is valid")
  }
}

impl CTXPopulationBoundryConditions {
  pub fn to_toml(&self, stim_py_qualified_name: &str) -> toml::Value {
    let mut table = toml::value::Table::new();
    table.insert("count".to_owned(), (self.count as i64).into());
    table.insert("stim".to_owned(), toml::Value::String(stim_py_qualified_name.to_owned()));
    toml::Value::Table(table)
  }
}

pub type BuilderCTXBoundary = Builder<CTX, Boundary, CTXPopulationBoundryConditions>;

impl Builder<CTX, Boundary, CTXPopulationBoundryConditions> {
  pub fn finish(self, ctx_count: usize) -> CTXPopulationBoundryConditions {
    let stimuli =
      toml_py_function_qualname_to_py_object(self.map.get("stimuli").expect("default should be set by caller"));

    CTXPopulationBoundryConditions { count: ctx_count, stimuli }
  }
}

pub type CTXConfig = NeuronConfig<CTX, CTXParameters, CTXPopulationBoundryConditions>;

impl Build<CTX, Boundary> for CTXPopulationBoundryConditions {
  const PYTHON_CALLABLE_FIELD_NAMES: &[&'static str] = &["stimuli"];
}

impl CTXPopulationBoundryConditions {}

fn p_si_given_sj(si: f64, sj: f64, sig_s: f64) -> f64 {
  1. / (1. + ((si - sj) / sig_s).powi(2))
}

fn stimuli_to_firing_rates(
  stimuli: &Array2<f64>,
  stimulated_rate: f64,
  base_rate: f64,
  sig_s: f64,
  max_rate: f64,
) -> Array2<f64> {
  // Assume stimuli position is distributed on one axis in a [0,1] feature spaces
  let stimuli_position = Array1::<f64>::linspace(0., 1., stimuli.ncols());
  let mut firing_rates = Array2::<f64>::from_elem(stimuli.raw_dim(), base_rate);

  for i in 0..firing_rates.ncols() {
    for j in 0..firing_rates.ncols() {
      ndarray::Zip::from(firing_rates.column_mut(i)).and(stimuli.column(j)).for_each(|rate, stim| {
        *rate += stim * p_si_given_sj(stimuli_position[i], stimuli_position[j], sig_s) * stimulated_rate;
      });
    }
  }
  firing_rates.mapv_inplace(|rate| rate.min(max_rate));

  firing_rates
}

fn firing_rates_to_cortical_spikes(firing_rates: &Array2<f64>, dt: f64) -> Array2<f64> {
  let time = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() as u64;
  let mut rng = rand_isaac::isaac64::Isaac64Rng::seed_from_u64(time);
  let uniform = Array2::<f64>::random_using(firing_rates.raw_dim(), Open01, &mut rng);
  let spikes = ndarray::Zip::from(firing_rates).and(&uniform).map_collect(|&r, &u| (u < r * dt) as i8 as f64);
  enforce_min_inter_spike_interval(&spikes, dt)
}

fn enforce_min_inter_spike_interval(spikes: &Array2<f64>, dt: f64) -> Array2<f64> {
  const MIN_ISI_MS: f64 = 2.;
  let mut isi_spikes = Array2::<f64>::zeros(spikes.raw_dim());
  let mut time_last_spike = Array1::<f64>::zeros(spikes.ncols());

  for t in 0..spikes.nrows() {
    let time_now = t as f64 * dt;
    for n in 0..spikes.ncols() {
      if spikes[[t, n]] == 1. && time_now - time_last_spike[n] >= MIN_ISI_MS {
        isi_spikes[[t, n]] = 1.;
        time_last_spike[n] = time_now;
      }
    }
  }

  isi_spikes
}

fn convolve_spike_train(spikes: &Array2<f64>, sigma_ray: f64, dt: f64, kernel_len: f64) -> Array2<f64> {
  let kernel_len = (kernel_len / dt) as usize;
  let kernel = rayleigh_kernel(sigma_ray, dt, kernel_len);
  let mut result = Array2::<f64>::zeros(spikes.raw_dim());

  for n in 0..spikes.ncols() {
    result.column_mut(n).assign(&convolve_1d(spikes.column(n), &kernel));
  }

  result
}

fn convolve_1d(signal: ArrayView1<f64>, kernel: &Array1<f64>) -> Array1<f64> {
  let n = signal.len();
  let k = kernel.len();
  let mut out = Array1::<f64>::zeros(n);

  for t in 0..n {
    let mut acc = 0.0;
    for ki in 0..k {
      if t >= ki {
        acc += signal[t - ki] * kernel[ki];
      }
    }
    out[t] = acc;
  }

  out
}

fn rayleigh_kernel(sigma_ray: f64, dt: f64, length: usize) -> Array1<f64> {
  Array1::from_iter((0..length).map(|i| {
    let t = i as f64 * dt;
    (t / (sigma_ray * sigma_ray)) * f64::exp(-t * t / (2. * sigma_ray * sigma_ray))
  }))
}
