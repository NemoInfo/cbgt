use ndarray::{array, Array2};
use numpy::{IntoPyArray, PyArray2, PyArrayMethods};
use pyo3::{ffi::c_str, prelude::*, types::PyDict};

use std::{collections::HashMap, path::PathBuf};

mod parameters;
use parameters::*;

mod stn;
use stn::STNPopulation;

mod gpe;
use gpe::GPePopulation;

mod util;

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
  /// Parameter file path
  pub parameters_file: PathBuf,
  /// Parameter file path
  pub parameters_settings: String,
  /// Number of neurons in STN
  #[pyo3(get)]
  pub num_stn: usize,
  /// Number of neurons in GPe
  #[pyo3(get)]
  pub num_gpe: usize,
  /// External STN current
  pub i_ext_stn: Array2<f64>,
  /// External GPe current
  pub i_ext_gpe: Array2<f64>,
  /// Subcortical simulated GPe current
  pub i_app_gpe: Array2<f64>,
  /// GPe -> STN connectivity matrix
  pub c_g_s: Array2<f64>,
  /// GPe -> GPe connectivity matrix
  pub c_g_g: Array2<f64>,
  /// STN -> GPe connectivity matrix
  pub c_s_g: Array2<f64>,
}

impl RubinTerman {
  pub fn new(num_stn: usize, num_gpe: usize, dt: f64, total_t: f64) -> Self {
    let num_timesteps: usize = (total_t * 1e3 / dt) as usize;

    RubinTerman {
      dt,
      total_t,
      num_stn,
      num_gpe,
      parameters_file: "src/PARAMETERS.toml".into(),
      parameters_settings: "default".to_owned(),
      i_ext_stn: Array2::zeros((num_timesteps, num_stn)),
      i_ext_gpe: Array2::zeros((num_timesteps, num_gpe)),
      i_app_gpe: Array2::zeros((num_timesteps, num_gpe)),
      c_g_s: Array2::zeros((num_gpe, num_stn)),
      c_g_g: Array2::zeros((num_gpe, num_gpe)),
      c_s_g: Array2::zeros((num_stn, num_gpe)),
    }
  }
}

impl RubinTerman {
  pub fn _run(&mut self) -> HashMap<&str, HashMap<&str, Array2<f64>>> {
    let n_timesteps: usize = (self.total_t * 1e3 / self.dt) as usize;
    let stn_parameters = STNParameters::from_config(&self.parameters_file, &self.parameters_settings);
    let gpe_parameters = GPeParameters::from_config(&self.parameters_file, &self.parameters_settings);

    let mut stn = STNPopulation::new(n_timesteps, self.num_stn, self.i_ext_stn.clone(), self.c_g_s.clone());
    let mut gpe = GPePopulation::new(
      n_timesteps,
      self.num_stn,
      self.i_ext_gpe.clone(),
      self.i_app_gpe.clone(),
      self.c_s_g.clone(),
      self.c_g_g.clone(),
    );

    stn.set_ics_from_config(&self.parameters_file, &self.parameters_settings);
    gpe.set_ics_from_config(&self.parameters_file, &self.parameters_settings);

    for it in 0..n_timesteps - 1 {
      stn.euler_step(it, self.dt, &stn_parameters, &gpe.s.row(it));
      gpe.euler_step(it, self.dt, &gpe_parameters, &stn.s.row(it));
    }

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

  pub fn vectorize_i_ext<F>(i_ext: F, dt: f64, total_t: f64, num_neurons: usize) -> Array2<f64>
  where
    F: Fn(f64, usize) -> f64,
  {
    let num_timesteps: usize = (total_t * 1e3 / dt) as usize;
    let mut a = Array2::<f64>::zeros((num_timesteps, num_neurons));
    for n in 0..num_neurons {
      for it in 0..num_timesteps {
        a[[it, n]] = i_ext(it as f64 * dt, n);
      }
    }
    a
  }

  pub fn vectorize_i_ext_py(i_ext_py: &PyObject, dt: f64, total_t: f64, num_neurons: usize) -> Array2<f64> {
    let num_timesteps: usize = (total_t * 1e3 / dt) as usize;
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
      let mut a = Array2::<f64>::zeros((num_timesteps, num_neurons));
      for n in 0..num_neurons {
        for it in 0..num_timesteps {
          a[[it, n]] = i_ext_py.call1(py, (it as f64 * dt, n)).unwrap().extract(py).unwrap();
        }
      }
      a
    })
  }
}

fn default_i_ext_py() -> PyObject {
  pyo3::prepare_freethreaded_python();
  Python::with_gil(|py| py.eval(c_str!("lambda t, n: 0.0"), None, None).unwrap().into())
}

fn pyarray_to_ndarray<T: numpy::Element>(arr: Py<PyArray2<T>>) -> Array2<T> {
  pyo3::prepare_freethreaded_python();
  Python::with_gil(|py| arr.bind(py).to_owned_array())
}

#[pymethods]
impl RubinTerman {
  #[new]
  #[pyo3(signature = (dt = 0.01, total_t = 2.0, parameters_file = PathBuf::from("src/PARAMETERS.toml"),  
      parameters_settings="default".to_owned(), num_stn=10, num_gpe=10, 
      i_ext_stn=default_i_ext_py(), i_ext_gpe=default_i_ext_py(), i_app_gpe=default_i_ext_py(),
      c_g_s=None, c_s_g=None, c_g_g=None)
    )]
  fn new_py(
    dt: f64,
    total_t: f64,
    parameters_file: PathBuf,
    parameters_settings: String,
    num_stn: usize,
    num_gpe: usize,
    i_ext_stn: PyObject,
    i_ext_gpe: PyObject,
    i_app_gpe: PyObject,
    c_g_s: Option<Py<PyArray2<f64>>>,
    c_s_g: Option<Py<PyArray2<f64>>>,
    c_g_g: Option<Py<PyArray2<f64>>>,
  ) -> Self {
    Self {
      dt,
      total_t,
      parameters_file,
      parameters_settings,
      num_stn,
      num_gpe,
      i_ext_stn: Self::vectorize_i_ext_py(&i_ext_stn, dt, total_t, num_stn),
      i_ext_gpe: Self::vectorize_i_ext_py(&i_ext_gpe, dt, total_t, num_gpe),
      i_app_gpe: Self::vectorize_i_ext_py(&i_app_gpe, dt, total_t, num_gpe),
      c_g_s: match c_g_s {
        None => Array2::zeros((num_gpe, num_stn)),
        Some(arr) => pyarray_to_ndarray(arr),
      },
      c_s_g: match c_s_g {
        None => Array2::zeros((num_gpe, num_stn)),
        Some(arr) => pyarray_to_ndarray(arr),
      },
      c_g_g: match c_g_g {
        None => Array2::zeros((num_gpe, num_stn)),
        Some(arr) => pyarray_to_ndarray(arr),
      },
    }
  }

  fn run(&mut self, py: Python) -> Py<PyDict> {
    let res = self._run();
    println!("Simulation completed!");
    to_python_dict(py, res)
  }
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
    let a = RubinTerman::vectorize_i_ext_py(
      &Python::with_gil(|py| py.eval(c_str!("lambda t, n: 6.9 if t < 500 else 9.6"), None, None).unwrap().into()),
      dt,
      total_t,
      num_neurons,
    );
    assert_eq!(a[[0, 0]], 6.9);
    assert_eq!(a[[a.shape()[0] - 1, 0]], 9.6);

    let a = RubinTerman::vectorize_i_ext_py(
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
    let a = RubinTerman::vectorize_i_ext(|t, _| if t < 500. { 6.9 } else { 9.6 }, 0.01, 1., 5);
    assert_eq!(a[[0, 0]], 6.9);
    assert_eq!(a[[a.shape()[0] - 1, 0]], 9.6);
    let a = RubinTerman::vectorize_i_ext(|t, _| if t < 500. { 6.9 } else { 9.6 }, 0.1, 1., 5);
    assert_eq!(a[[0, 0]], 6.9);
    assert_eq!(a[[a.shape()[0] - 1, 0]], 9.6);
  }

  #[test]
  fn test_load_ics() {
    let rt = RubinTerman { parameters_settings: "test".to_string(), ..RubinTerman::new(10, 10, 0.01, 2.) };
    let mut stn = STNPopulation::new(10, rt.num_stn, rt.i_ext_stn.clone(), rt.c_g_s.clone());
    stn.set_ics_from_config(&rt.parameters_file, &rt.parameters_settings);
    assert_eq!(stn.h[[0, 0]], -69.);
    assert_eq!(stn.v[[0, 9]], -59.20177081754847);
  }
}
