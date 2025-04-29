use log::{debug, info};
use pyo3::{prelude::*, types::PyDict};
use struct_field_names_as_array::FieldNamesAsSlice;

use std::str::FromStr;
use std::sync::Arc;
use std::time::Instant;
use std::{collections::HashMap, thread};

mod parameters;
use parameters::*;

mod stn;
use stn::{BuilderSTNBoundary, STNPopulation, STNPopulationBoundryConditions, STN};

mod gpe;
use gpe::{BuilderGPeBoundary, GPe, GPePopulation, GPePopulationBoundryConditions};

mod util;
use util::*;

mod types;
use types::{Parameters, *};

#[pyclass]
/// Rubin Terman model using Euler's method
pub struct RubinTerman {
  /// Time between Euler steps (ms)
  #[pyo3(get)]
  pub dt: f64,
  /// Simulation time (s)
  #[pyo3(get)]
  pub total_t: f64,
  pub stn_population: STNPopulation,
  pub stn_parameters: STNParameters,
  pub gpe_population: GPePopulation,
  pub gpe_parameters: GPeParameters,
}

impl RubinTerman {
  pub fn new(
    dt: f64,
    total_t: f64,
    experiment: Option<(&str, Option<&str>)>,
    parameters_file: Option<&str>,
    boundry_ic_file: Option<&str>,
    use_default: bool,
    save_dir: Option<String>,
  ) -> Self {
    let (experiment, experiment_version) = experiment.map_or((None, None), |(e, v)| (Some(e), v));

    Self::new_py(
      dt,
      total_t,
      experiment,
      experiment_version,
      parameters_file,
      boundry_ic_file,
      use_default,
      save_dir,
      None,
    )
  }

  pub fn run(&mut self) {
    let num_timesteps: usize = (self.total_t / self.dt) as usize;

    debug!("Computing {} timesteps", format_number(num_timesteps));

    let stn_parameters = self.stn_parameters.clone();
    let gpe_parameters = self.gpe_parameters.clone();
    let mut stn = self.stn_population.clone();
    let mut gpe = self.gpe_population.clone();

    //debug!("{} STN neurons", self.stn_population.count);
    //debug!("{} GPe neurons", self.num_gpe);
    debug!("STN -> GPe\n{:?}", gpe.w_s_g.mapv(|x| x as isize));
    debug!("GPe -> GPe\n{:?}", gpe.w_g_g.mapv(|x| x as isize));
    debug!("GPe -> STN\n{:?}", stn.w_g_s.mapv(|x| x as isize));

    let spin_barrier = Arc::new(SpinBarrier::new(2));

    let dt = self.dt;
    // Hacky way to keep a view of the synapse in the other thread.
    // This is okay as long as nuclei state integration is time-synchronised between threads.
    // Since euler_step only modifies _.s.row(it + 1) we can safely read _.s.row(it) at the same time
    let gpe_s = unsafe { ndarray::ArrayView2::from_shape_ptr(gpe.s.raw_dim(), gpe.s.as_ptr()) };
    let gpe_rho_pre = unsafe { ndarray::ArrayView2::from_shape_ptr(gpe.rho_pre.raw_dim(), gpe.rho_pre.as_ptr()) };
    let stn_s = unsafe { ndarray::ArrayView2::from_shape_ptr(stn.s.raw_dim(), stn.s.as_ptr()) };
    let stn_rho_pre = unsafe { ndarray::ArrayView2::from_shape_ptr(stn.rho_pre.raw_dim(), stn.rho_pre.as_ptr()) };

    /* STN THREAD */
    let start = Instant::now();
    let barrier = spin_barrier.clone();
    let stn_thread = thread::spawn(move || {
      let start = Instant::now();
      for it in 0..num_timesteps - 1 {
        barrier.wait();
        stn.euler_step(it, dt, &stn_parameters, &gpe_s.row(it), &gpe_rho_pre.row(it));
      }
      debug!("STN time: {:.2}s", start.elapsed().as_secs_f64());
      stn
    });

    /* GPe THREAD */
    let barrier = spin_barrier;
    let gpe_thread = thread::spawn(move || {
      let start = Instant::now();
      for it in 0..num_timesteps - 1 {
        barrier.wait();
        gpe.euler_step(it, dt, &gpe_parameters, &stn_s.row(it), &stn_rho_pre.row(it));
      }
      debug!("GPe time: {:.2}s", start.elapsed().as_secs_f64());
      gpe
    });

    self.stn_population = stn_thread.join().expect("STN thread panicked!");
    self.gpe_population = gpe_thread.join().expect("GPe thread panicked!");

    info!("STN -> GPe:\n{}", self.stn_population.w_g_s);
    info!("GPe -> GPe:\n{}", self.gpe_population.w_g_g);
    info!("GPe -> STN:\n{}", self.gpe_population.w_s_g);

    info!(
      "Total real time: {:.2}s at {:.0} ms/sim_s",
      start.elapsed().as_secs_f64(),
      start.elapsed().as_secs_f64() / self.total_t
    );
  }

  pub fn into_map_polars_dataframe(self, output_dt: Option<f64>) {
    let stn = self.stn_population.into_compressed_polars_df(self.dt, output_dt);
    let gpe = self.gpe_population.into_compressed_polars_df(self.dt, output_dt);
    debug!("{:?}", gpe);

    let _: HashMap<&str, polars::prelude::DataFrame> = HashMap::from_iter([("stn", stn), ("gpe", gpe)]);
  }
}

pub const TMP_PYF_FILE_PATH: &'static str = ".";
pub const TMP_PYF_FILE_NAME: &'static str = "temp_functions";
pub const PYF_FILE_NAME: &'static str = "functions";

struct NeuronConfig<N, ParameterBuilder, BoundaryBuilder>
where
  N: Neuron,
  ParameterBuilder: Build<N, Parameters> + FieldNamesAsSlice,
  BoundaryBuilder: Build<N, Boundary> + FieldNamesAsSlice,
{
  par_map: toml::map::Map<String, toml::Value>,
  bcs_map: toml::map::Map<String, toml::Value>,
  pyf_map: HashMap<String, String>,
  _marker: std::marker::PhantomData<(N, ParameterBuilder, BoundaryBuilder)>,
}
// TODO: Test some invalid input

impl<N, ParameterBuilder, BoundaryBuilder> NeuronConfig<N, ParameterBuilder, BoundaryBuilder>
where
  N: Neuron,
  ParameterBuilder: Build<N, Parameters> + FieldNamesAsSlice,
  BoundaryBuilder: Build<N, Boundary> + FieldNamesAsSlice,
{
  fn new() -> Self {
    Self {
      par_map: toml::map::Map::new(),
      bcs_map: toml::map::Map::new(),
      pyf_map: HashMap::new(),
      _marker: std::marker::PhantomData,
    }
  }

  fn update_from_py(&mut self, key: &Bound<'_, PyAny>, value: &Bound<'_, PyAny>) -> bool {
    if let Some(key) = key.to_string().strip_prefix(&format!("{}_", N::TYPE.to_lowercase())) {
      if ParameterBuilder::FIELD_NAMES_AS_SLICE.contains(&key) {
        let kv = parse_toml_value(key, &format!("{value:?}")).as_table().unwrap().clone();
        self.par_map.extend(kv);
        return true;
      } else if BoundaryBuilder::FIELD_NAMES_AS_SLICE.contains(&key) {
        let kv = self.parse_toml_callable_py(key, value);
        self.bcs_map.extend(kv);
        return true;
      }
    }
    false
  }

  fn parse_toml_callable_py(&mut self, key: &str, value: &Bound<'_, PyAny>) -> toml::map::Map<String, toml::Value> {
    if BoundaryBuilder::PYTHON_CALLABLE_FIELD_NAMES.contains(&key) {
      let (src, name) =
        get_py_function_source_and_name(value.as_unbound()).expect("Could not get source and name of function");
      let qname = format!("{TMP_PYF_FILE_PATH}/{TMP_PYF_FILE_NAME}.{name}");
      self.pyf_map.insert(qname.clone(), src);
      toml::map::Map::from_iter([(key.into(), toml::Value::String(qname))])
    } else {
      parse_toml_value(key, &format!("{value:?}")).try_into().unwrap()
    }
  }

  fn into_maps(
    self,
    pyf_src: &mut HashMap<String, String>,
  ) -> (toml::map::Map<String, toml::Value>, toml::map::Map<String, toml::Value>) {
    pyf_src.extend(self.pyf_map);
    (self.par_map, self.bcs_map)
  }
}

type STNConfig = NeuronConfig<STN, STNParameters, STNPopulationBoundryConditions>;
type GPeConfig = NeuronConfig<GPe, GPeParameters, GPePopulationBoundryConditions>;

#[pymethods]
impl RubinTerman {
  #[new]
  #[pyo3(signature=(dt=0.01, total_t=2., experiment=None, experiment_version=None, 
                    parameters_file=None, boundry_ic_file=None, use_default=true,
                    save_dir=Some("/tmp/cbgt_last_model".to_owned()), **kwds))]
  fn new_py(
    dt: f64,
    mut total_t: f64,
    experiment: Option<&str>,
    experiment_version: Option<&str>,
    parameters_file: Option<&str>,
    boundry_ic_file: Option<&str>,
    use_default: bool,
    save_dir: Option<String>, // Maybe add datetime to temp save
    kwds: Option<&Bound<'_, PyDict>>,
  ) -> Self {
    total_t *= 1e3;
    assert!(experiment.is_some() || experiment_version.is_none(), "Experiment version requires experiment");

    let save_dir = save_dir.map(|x| x.trim_end_matches("/").to_owned());
    if let Some(save_dir) = &save_dir {
      std::fs::create_dir_all(save_dir).expect("Could not create save folder");
    }

    let experiment = experiment.map(|name| (name, experiment_version));

    let mut stn_kw_config = STNConfig::new();
    let mut gpe_kw_config = GPeConfig::new();

    if let Some(kwds) = kwds {
      for (key, val) in kwds {
        if !(stn_kw_config.update_from_py(&key, &val) || gpe_kw_config.update_from_py(&key, &val)) {
          panic!("Unexpected key word argument {}", key);
        }
      }
    }

    let mut pyf_src = HashMap::from_iter(match Boundary::DEFAULT {
      Some((qname, src)) => vec![(qname.into(), src.into())],
      None => vec![],
    });
    let (stn_kw_params, stn_kw_bcs) = stn_kw_config.into_maps(&mut pyf_src);
    let (gpe_kw_params, gpe_kw_bcs) = gpe_kw_config.into_maps(&mut pyf_src);

    let stn_parameters = BuilderSTNParameters::build(stn_kw_params, parameters_file, experiment, use_default).finish();
    let gpe_parameters = BuilderGPeParameters::build(gpe_kw_params, parameters_file, experiment, use_default).finish();

    log::debug!("{:?}", stn_kw_bcs);
    let stn_bcs_builder = BuilderSTNBoundary::build(stn_kw_bcs, boundry_ic_file, experiment, use_default);
    let gpe_bcs_builder = BuilderGPeBoundary::build(gpe_kw_bcs, boundry_ic_file, experiment, use_default);

    log::debug!("{:?}", stn_bcs_builder);
    let stn_count = stn_bcs_builder.get_count().expect("stn_count not found");
    let gpe_count = gpe_bcs_builder.get_count().expect("gpe_count not found");

    stn_bcs_builder.extends_pyf_src(&mut pyf_src);
    gpe_bcs_builder.extends_pyf_src(&mut pyf_src);

    let stn_qual_names = stn_bcs_builder.get_callable_qnames();
    let gpe_qual_names = gpe_bcs_builder.get_callable_qnames();
    let pyf_file = write_temp_pyf_file(pyf_src);

    let num_timesteps: usize = (total_t / dt) as usize;
    let stn_bcs = stn_bcs_builder.finish(stn_count, gpe_count, dt, total_t);
    let gpe_bcs = gpe_bcs_builder.finish(gpe_count, stn_count, dt, total_t);

    if let Some(save_dir) = &save_dir {
      write_parameter_file(&stn_parameters, &gpe_parameters, save_dir);
      write_boundary_file(&stn_bcs, &gpe_bcs, save_dir, &stn_qual_names, &gpe_qual_names);
      std::fs::copy(&pyf_file, format!("{save_dir}/{PYF_FILE_NAME}.py")).unwrap();
    }

    let stn_population = STNPopulation::new(num_timesteps, stn_count, gpe_count).with_bcs(stn_bcs);
    let gpe_population = GPePopulation::new(num_timesteps, stn_count, gpe_count).with_bcs(gpe_bcs);

    std::fs::remove_file(pyf_file).expect("File was just created, it should exist.");

    Self { dt, total_t, stn_population, stn_parameters, gpe_population, gpe_parameters }
  }

  #[pyo3(name = "run")]
  fn run_py(&mut self) {
    self.run();
  }

  #[pyo3(name="to_polars", signature = (dt=None))]
  fn into_map_polars_dataframe_py(&self, py: Python, dt: Option<f64>) -> Py<PyDict> {
    let stn = self.stn_population.into_compressed_polars_df(self.dt, dt);
    let gpe = self.gpe_population.into_compressed_polars_df(self.dt, dt);

    let dict = PyDict::new(py);
    dict.set_item("stn", pyo3_polars::PyDataFrame(stn)).expect("Could not add insert STN Polars DataFrame");
    dict.set_item("gpe", pyo3_polars::PyDataFrame(gpe)).expect("Could not add insert STN Polars DataFrame");

    dict.into()
  }

  #[pyo3(signature = (dir, dt=None))]
  fn save_to_parquet_files(&self, dir: &str, dt: Option<f64>) -> PyResult<()> {
    let dir = std::path::PathBuf::from_str(dir)?;
    std::fs::create_dir_all(&dir)?;

    let mut stn = self.stn_population.into_compressed_polars_df(self.dt, dt);
    let mut gpe = self.gpe_population.into_compressed_polars_df(self.dt, dt);

    let write_parquet = |name: &str, df: &mut polars::prelude::DataFrame| {
      let file_path = dir.join(name);
      let file = std::fs::File::create(&file_path).expect("Could not creare output file");
      let writer = polars::prelude::ParquetWriter::new(&file);
      writer.set_parallel(true).finish(df).expect("Could not write to output file");
      file_path
    };

    write_parquet("stn.parquet", &mut stn);
    write_parquet("gpe.parquet", &mut gpe);

    Ok(())
  }

  #[staticmethod]
  #[pyo3(signature = (log_level="debug"))]
  fn init_logger(log_level: &str) {
    _ = env_logger::Builder::new().filter_level(log_level.parse().expect("Invalid log level")).try_init();
  }
}

#[pymodule]
fn cbgt(m: &Bound<'_, PyModule>) -> PyResult<()> {
  m.add_class::<RubinTerman>()?;
  Ok(())
}

#[cfg(test)]
mod rubin_terman {
  use super::*;
  use pyo3::ffi::c_str;
  use pyo3::Python;

  #[test]
  fn test_vectorize_i_ext_py() {
    pyo3::prepare_freethreaded_python();
    let dt = 0.01;
    let total_t = 1. * 1e3;
    let num_neurons = 5;
    let a = vectorize_i_ext_py(
      &Python::with_gil(|py| py.eval(c_str!("lambda t, n: 6.9 if t < 500 else 9.6"), None, None).unwrap().into()),
      dt,
      total_t,
      num_neurons,
    );
    assert_eq!(a[[0, 0]], 6.9);
    assert_eq!(a[[a.shape()[0] - 1, 0]], 9.6);

    let a = vectorize_i_ext_py(
      &Python::with_gil(|py| py.eval(c_str!("lambda t, n: 6.9 if t < 500 else 9.6"), None, None).unwrap().into()),
      0.1,
      total_t,
      num_neurons,
    );
    assert_eq!(a[[0, 0]], 6.9);
    assert_eq!(a[[a.shape()[0] - 1, 0]], 9.6);
  }

  #[test]
  fn test_vectorize_i_ext_rust() {
    let a = vectorize_i_ext(|t, _| if t < 500. { 6.9 } else { 9.6 }, 0.01, 1e3, 5);
    assert_eq!(a[[0, 0]], 6.9);
    assert_eq!(a[[a.shape()[0] - 1, 0]], 9.6);
    let a = vectorize_i_ext(|t, _| if t < 500. { 6.9 } else { 9.6 }, 0.1, 1e3, 5);
    assert_eq!(a[[0, 0]], 6.9);
    assert_eq!(a[[a.shape()[0] - 1, 0]], 9.6);
  }
}
