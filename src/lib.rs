use log::{debug, info};
use ndarray::{Array2, ArrayView2};
use numpy::IntoPyArray;
use pyo3::{prelude::*, types::PyDict};
use struct_field_names_as_array::FieldNamesAsArray;

use std::io::Write;
use std::sync::Arc;
use std::time::Instant;
use std::{collections::HashMap, thread};

mod parameters;
use parameters::*;

mod stn;
use stn::{STNPopulation, STNPopulationBoundryConditions};

mod gpe;
use gpe::{GPePopulation, GPePopulationBoundryConditions};

mod util;
use util::*;

/// Rubin Terman model using Euler's method
#[allow(unused)]
#[pyclass]
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
    stn_count: usize,
    gpe_count: usize,
    dt: f64,
    total_t: f64,
    use_default: bool,
    experiment: Option<(&str, Option<&str>)>,
    custom_file: Option<&str>,
  ) -> Self {
    let num_timesteps: usize = (total_t * 1e3 / dt) as usize;

    let custom_stn = custom_file.map(|f| read_map_from_toml(f, None, "STN"));
    let custom_gpe = custom_file.map(|f| read_map_from_toml(f, None, "GPe"));

    let stn_bcs = STNPopulationBoundryConditions::from(
      STNPopulationBoundryConditions::build_map(use_default, experiment, custom_stn.clone()),
      stn_count,
      gpe_count,
      dt,
      total_t,
    );

    let gpe_bcs = GPePopulationBoundryConditions::from(
      GPePopulationBoundryConditions::build_map(use_default, experiment, custom_gpe.clone()),
      gpe_count,
      stn_count,
      dt,
      total_t,
    );

    Self {
      dt,
      total_t,
      stn_population: STNPopulation::new(num_timesteps, stn_count, gpe_count).with_bcs(stn_bcs),
      stn_parameters: STNParameters::build(use_default, experiment, custom_stn),
      gpe_population: GPePopulation::new(num_timesteps, gpe_count, stn_count).with_bcs(gpe_bcs),
      gpe_parameters: GPeParameters::build(use_default, experiment, custom_gpe),
    }
  }
}

impl RubinTerman {
  pub fn _run(&mut self) -> HashMap<&str, HashMap<&str, Array2<f64>>> {
    let num_timesteps: usize = (self.total_t * 1e3 / self.dt) as usize;

    debug!("Computing {} timesteps", format_number(num_timesteps));

    let stn_parameters = self.stn_parameters.clone();
    let gpe_parameters = self.gpe_parameters.clone();
    let mut stn = self.stn_population.clone();
    let mut gpe = self.gpe_population.clone();

    //debug!("{} STN neurons", self.stn_population.count);
    //debug!("{} GPe neurons", self.num_gpe);
    debug!("STN -> GPe\n{:?}", gpe.c_s_g.mapv(|x| x as isize));
    debug!("GPe -> GPe\n{:?}", gpe.c_g_g.mapv(|x| x as isize));
    debug!("GPe -> STN\n{:?}", stn.c_g_s.mapv(|x| x as isize));

    let spin_barrier = Arc::new(SpinBarrier::new(2));

    let dt = self.dt;
    // Hacky way to keep a view of the synapse in the other thread.
    // This is okay as long as nuclei state integration is time-synchronised between threads.
    // Since euler_step only modifies _.s.row(it + 1) we can safely read _.s.row(it) at the same time
    let gpe_s = unsafe { ArrayView2::from_shape_ptr(gpe.s.raw_dim(), gpe.s.as_ptr()) };
    let stn_s = unsafe { ArrayView2::from_shape_ptr(stn.s.raw_dim(), stn.s.as_ptr()) };

    let start = Instant::now();
    let barrier = spin_barrier.clone();
    let stn_thread = thread::spawn(move || {
      let start = Instant::now();
      for it in 0..num_timesteps - 1 {
        barrier.wait();
        stn.euler_step(it, dt, &stn_parameters, &gpe_s.row(it));
      }
      debug!("STN time: {:.2}s", start.elapsed().as_secs_f64());
      stn
    });

    let barrier = spin_barrier;
    let gpe_thread = thread::spawn(move || {
      let start = Instant::now();
      for it in 0..num_timesteps - 1 {
        barrier.wait();
        gpe.euler_step(it, dt, &gpe_parameters, &stn_s.row(it));
      }
      debug!("GPe time: {:.2}s", start.elapsed().as_secs_f64());
      gpe
    });

    let stn = stn_thread.join().expect("STN thread panicked!");
    let gpe = gpe_thread.join().expect("GPe thread panicked!");

    info!(
      "Total real time: {:.2}s at {:.0} ms/sim_s",
      start.elapsed().as_secs_f64(),
      start.elapsed().as_secs_f64() / self.total_t * 1e3
    );

    #[rustfmt::skip]
    let combined = HashMap::<&str, HashMap<&str, Array2<f64>>>::from([
      ("stn", HashMap::from([
				("v", stn.v), 
				("i_l", stn.i_l), 
				("i_k", stn.i_k), 
				("i_na", stn.i_na), 
				("i_t", stn.i_t), 
				("i_ca", stn.i_ca), 
				("i_ahp", stn.i_ahp), 
				("i_g_s", stn.i_g_s), 
				("i_ext", stn.i_ext), 
				("s", stn.s), 
			])),
      ("gpe", HashMap::from([
				("v", gpe.v), 
				("i_l", gpe.i_l), 
				("i_k", gpe.i_k), 
				("i_na", gpe.i_na), 
				("i_t", gpe.i_t), 
				("i_ca", gpe.i_ca), 
				("i_ahp", gpe.i_ahp), 
				("i_ext", gpe.i_ext), 
				("i_app", gpe.i_app), 
				("i_s_g", gpe.i_s_g), 
				("i_g_g", gpe.i_g_g), 
				("s", gpe.s), 
			])),
    ]);

    combined
  }
}

#[pymethods]
impl RubinTerman {
  #[pyo3(signature=(dt=0.01, total_t=2., experiment=None, experiment_version=None, 
                    parameters_file=None, boundry_ic_file=None, use_default=true,
                    stn_i_ext=None, gpe_i_ext=None, gpe_i_app=None, save_dir=Some("/tmp/cbgt_last_model".to_owned()), **kwds))]
  #[new]
  fn new_py(
    dt: f64,
    total_t: f64,
    experiment: Option<&str>,
    experiment_version: Option<&str>,
    parameters_file: Option<&str>,
    boundry_ic_file: Option<&str>,
    use_default: bool,
    stn_i_ext: Option<PyObject>,
    gpe_i_ext: Option<PyObject>,
    gpe_i_app: Option<PyObject>,
    save_dir: Option<String>, // Maybe add datetime to temp save
    kwds: Option<&Bound<'_, PyDict>>,
  ) -> Self {
    env_logger::init();
    assert!(experiment.is_some() || experiment_version.is_none(), "Experiment version requires experiment");

    if let Some(save_dir) = &save_dir {
      std::fs::create_dir_all(save_dir).expect("Could not create save folder");
    }

    let experiment = experiment.map(|name| (name, experiment_version));

    let mut map_stn_params = toml::map::Map::new();
    let mut map_gpe_params = toml::map::Map::new();
    let mut map_stn_bcs = toml::map::Map::new();
    let mut map_gpe_bcs = toml::map::Map::new();

    // Parse custom parameter/boundry keywords
    if let Some(kwds) = kwds {
      for (key, v) in kwds {
        if let Some(k) = key.to_string().strip_prefix("stn_") {
          if STNParameters::FIELD_NAMES_AS_ARRAY.contains(&k) {
            let kv: toml::Value = format!("{}={}", k.to_string(), v.to_string()).parse().unwrap();
            update_toml_map(&mut map_stn_params, kv.as_table().unwrap().to_owned());
          } else if STNPopulationBoundryConditions::FIELD_NAMES_AS_ARRAY.contains(&k) {
            let kv: toml::Value = parse_toml_value(k, &format!("{v:?}"));
            update_toml_map(&mut map_stn_bcs, kv.as_table().unwrap().to_owned());
          } else {
            panic!("Unrecognized kwarg {key}");
          }
        } else if let Some(k) = key.to_string().strip_prefix("gpe_") {
          if GPeParameters::FIELD_NAMES_AS_ARRAY.contains(&k) {
            let kv: toml::Value = format!("{}={}", k.to_string(), v.to_string()).parse().unwrap();
            update_toml_map(&mut map_gpe_params, kv.as_table().unwrap().to_owned());
          } else if GPePopulationBoundryConditions::FIELD_NAMES_AS_ARRAY.contains(&k) {
            let kv: toml::Value = parse_toml_value(k, &format!("{v:?}"));
            update_toml_map(&mut map_gpe_bcs, kv.as_table().unwrap().to_owned());
          } else {
            panic!("Unrecognized kwarg {key}");
          }
        } else {
          panic!("Unrecognized kwarg {key}");
        }
      }
    }

    assert!(
      parameters_file.is_none() || (map_stn_params.is_empty() && map_gpe_params.is_empty()),
      "Cannot give parameter both by keyword and by file at the same time!"
    );

    if let Some(file_path) = parameters_file {
      map_stn_params = read_map_from_toml(file_path, None, "STN");
      map_gpe_params = read_map_from_toml(file_path, None, "GPe");
    }

    let stn_parameters = STNParameters::build(use_default, experiment, Some(map_stn_params));
    let gpe_parameters = GPeParameters::build(use_default, experiment, Some(map_gpe_params));

    if let Some(save_dir) = &save_dir {
      let parameter_map = toml::value::Table::from_iter([
        ("STN".to_owned(), stn_parameters.to_toml()),
        ("GPe".to_owned(), gpe_parameters.to_toml()),
      ]);

      let file_path: std::path::PathBuf = format!("{save_dir}/{EXPERIMENT_PARAMETER_FILE_NAME}").into();
      let mut file = std::fs::File::create(&file_path).unwrap();
      write!(file, "{}", toml::Value::Table(parameter_map)).unwrap();
      info!("Saved parameters at [{}]", file_path.canonicalize().unwrap().display());
    }

    if let Some(file_path) = boundry_ic_file {
      map_stn_bcs = read_map_from_toml(file_path, None, "STN");
      map_gpe_bcs = read_map_from_toml(file_path, None, "GPe");
    }

    let stn_count = map_stn_bcs
      .get("count")
      .expect("STN Population count not found")
      .as_integer()
      .expect("integer")
      .try_into()
      .expect("usize");

    let gpe_count = map_gpe_bcs
      .get("count")
      .expect("GPe Population count not found")
      .as_integer()
      .expect("integer")
      .try_into()
      .expect("usize");

    let num_timesteps: usize = (total_t * 1e3 / dt) as usize;

    let mut function_sources: HashMap<String, String> = HashMap::new();

    if let Some(stn_i_ext) = stn_i_ext {
      // Maybe this should be a wraning
      let (src, name) = get_py_function_source_and_name(&stn_i_ext).expect("Could not get source and name of function");
      function_sources.insert(name.clone(), src);
      map_stn_bcs
        .insert("i_ext".to_owned(), toml::Value::String(format!("{}/functions.{name}", save_dir.as_ref().unwrap())));
      // Maybe create the module from source directly and only optionally saving the file
    }

    if let Some(gpe_i_ext) = gpe_i_ext {
      // Maybe this should be a wraning
      let (src, name) = get_py_function_source_and_name(&gpe_i_ext).expect("Could not get source and name of function");
      function_sources.insert(name.clone(), src);
      map_gpe_bcs
        .insert("i_ext".to_owned(), toml::Value::String(format!("{}/functions.{name}", save_dir.as_ref().unwrap())));
      // Bad unwrap
    }

    if let Some(gpe_i_app) = gpe_i_app {
      // Maybe this should be a wraning
      let (src, name) = get_py_function_source_and_name(&gpe_i_app).expect("Could not get source and name of function");
      function_sources.insert(name.clone(), src);
      map_gpe_bcs
        .insert("i_app".to_owned(), toml::Value::String(format!("{}/functions.{name}", save_dir.as_ref().unwrap())));
      // Bad unwrap
    }

    if let Some(save_dir) = &save_dir {
      let file_path = format!("{save_dir}/functions.py");
      std::fs::File::create(&file_path).unwrap();
      let mut file = std::fs::OpenOptions::new().append(true).open(&file_path).unwrap();
      info!("Saved functions  at [{save_dir}/functions.py]");

      for src in function_sources.values() {
        writeln!(file, "{src}").unwrap();
      }
    }

    let stn_bcs = STNPopulationBoundryConditions::from(map_stn_bcs, stn_count, gpe_count, dt, total_t);
    let stn_population = STNPopulation::new(num_timesteps, stn_count, gpe_count).with_bcs(stn_bcs);

    let gpe_bcs = GPePopulationBoundryConditions::from(map_gpe_bcs, gpe_count, stn_count, dt, total_t);
    let gpe_population = GPePopulation::new(num_timesteps, stn_count, gpe_count).with_bcs(gpe_bcs);

    // TODO Save bcs

    Self { dt, total_t, stn_population, stn_parameters, gpe_population, gpe_parameters }
  }

  fn run(&mut self, py: Python) -> Py<PyDict> {
    let res = self._run();
    to_python_dict(py, res)
  }
}

fn get_py_function_source_and_name(f: &PyObject) -> Option<(String, String)> {
  let src = get_py_function_source(&f)?;
  let mut name = get_py_object_name(&f)?;
  if name == "<lambda>" {
    name = src.split_once('=').unwrap().0.trim().to_owned();
  }

  Some((src, name))
}

fn to_python_dict<'py>(py: Python, rust_map: HashMap<&str, HashMap<&str, Array2<f64>>>) -> Py<PyDict> {
  let py_dict = PyDict::new(py);

  for (key1, sub_map) in rust_map {
    let sub_dict = PyDict::new(py);

    for (key2, array) in sub_map {
      sub_dict.set_item(key2, array.into_pyarray(py)).unwrap();
    }

    py_dict.set_item(key1, sub_dict).unwrap();
  }

  py_dict.into()
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
    let total_t = 1.;
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
    let a = vectorize_i_ext(|t, _| if t < 500. { 6.9 } else { 9.6 }, 0.01, 1., 5);
    assert_eq!(a[[0, 0]], 6.9);
    assert_eq!(a[[a.shape()[0] - 1, 0]], 9.6);
    let a = vectorize_i_ext(|t, _| if t < 500. { 6.9 } else { 9.6 }, 0.1, 1., 5);
    assert_eq!(a[[0, 0]], 6.9);
    assert_eq!(a[[a.shape()[0] - 1, 0]], 9.6);
  }

  // #[test]
  // fn test_load_ics() {
  //   let rt = RubinTerman::new(10, 10, 0.02, 2., true, None, None);
  //   dbg!(rt.stn_population.v.row(0));
  //   assert!(false);
  // }

  #[test]
  fn get_py_function_source() {
    assert!(false);
  }
}
