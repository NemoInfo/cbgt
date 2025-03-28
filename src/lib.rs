use ndarray::{array, Array2};
use numpy::IntoPyArray;
use pyo3::{ffi::c_str, prelude::*, types::PyDict};

use std::{collections::HashMap, path::PathBuf};

mod parameters;
use parameters::*;

mod stn;
use stn::STNPopulation;

mod util;

/// Rubin Terman model using Euler's method
#[allow(unused)]
#[pyclass(get_all)]
pub struct RubinTerman {
  /// Time between Euler steps (ms)
  pub dt: f64,
  /// Simulation time (s)
  pub total_t: f64,
  /// Parameter file path
  pub parameters_file: PathBuf,
  /// Parameter file path
  pub parameters_settings: String,
  /// Number of neurons in STN
  pub num_stn: usize,
  /// Number of neurons in GPe
  pub num_gpe: usize,
  /// External STN current (Python function)
  /// (time_ms: f64, neuron_idx: usize) -> f64
  pub i_ext_stn_py: PyObject,
  /// External GPe current (Python function)
  /// (time_ms: f64, neuron_idx: usize) -> f64
  pub i_ext_gpe_py: PyObject,
  /// External GPe current (Python function)
  /// (time_ms: f64, neuron_idx: usize) -> f64
  pub i_app_gpe_py: PyObject,
}

impl Default for RubinTerman {
  fn default() -> Self {
    RubinTerman {
      dt: 0.01,
      total_t: 2.0,
      parameters_file: "src/PARAMETERS.toml".into(),
      parameters_settings: "default".to_owned(),
      num_stn: 10,
      num_gpe: 10,
      i_ext_stn_py: default_i_ext_py(),
      i_ext_gpe_py: default_i_ext_py(),
      i_app_gpe_py: default_i_ext_py(),
    }
  }
}

impl RubinTerman {
  pub fn _run(&self) -> HashMap<&str, HashMap<&str, Array2<f64>>> {
    let n_timesteps: usize = (self.total_t * 1e3 / self.dt) as usize;
    let stn_parameters = STNParameters::from_config(&self.parameters_file, &self.parameters_settings);
    let _gpe_parameters = GPeParameters::from_config(&self.parameters_file, &self.parameters_settings);

    let i_ext_stn = self.vectorize_i_ext(&self.i_ext_stn_py);
    let mut stn = STNPopulation::new(n_timesteps, self.num_stn, i_ext_stn);

    // Create GPe currents
    let _i_ext_gpe = self.vectorize_i_ext(&self.i_ext_gpe_py);
    let _i_app_gpe = self.vectorize_i_ext(&self.i_app_gpe_py);

    stn.v.row_mut(0).assign(&array![
      -59.62828421888404,
      -61.0485669306943,
      -59.9232859246653,
      -58.70506521874258,
      -59.81316532105502,
      -60.41737514151719,
      -60.57000688576042,
      -60.77581472006873,
      -59.72163362685856,
      -59.20177081754847
    ]);
    stn.h.row_mut(0).assign(&array![
      0.5063486245631907,
      0.2933274739456392,
      0.4828268896903307,
      0.5957938758715363,
      0.4801708406464686,
      0.397555659151211,
      0.3761635970127477,
      0.3316364917935809,
      0.4881964058107033,
      0.5373898124788108
    ]);
    stn.n.row_mut(0).assign(&array![
      0.0301468039831072,
      0.04412485475791555,
      0.02936940165051648,
      0.03307223867110721,
      0.02961425249063069,
      0.02990618866753074,
      0.03096707115136645,
      0.03603641291454053,
      0.02983123244237023,
      0.03137696787429014
    ]);
    stn.r.row_mut(0).assign(&array![
      0.0295473069771012,
      0.07318677802595788,
      0.03401991571903244,
      0.01899268957583912,
      0.0322092810112401,
      0.04490215539151968,
      0.0496024428039565,
      0.05982606979469521,
      0.03078507359379932,
      0.02403333448524015
    ]);
    stn.ca.row_mut(0).assign(&array![
      0.2994323366425385,
      0.4076730264403847,
      0.3271760563827424,
      0.2456039126383157,
      0.3090126869287847,
      0.3533066857313201,
      0.3668697913124569,
      0.3777575381495549,
      0.3008309498107221,
      0.2631312497961643
    ]);
    stn.s.row_mut(0).assign(&array![
      0.008821617722180833,
      0.007400276913597601,
      0.00850582621763913,
      0.009886276645187469,
      0.00862235586166425,
      0.008001611992658621,
      0.007851916739337694,
      0.007654426383227644,
      0.008720434017133022,
      0.009298664650592724
    ]);

    for it in 0..n_timesteps - 1 {
      stn.euler_step(it, self.dt, &stn_parameters);
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
				("s", stn.s), 
			])),
    ]);

    combined
  }

  pub fn vectorize_i_ext(&self, i_ext_py: &PyObject) -> Array2<f64> {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
      let n_timesteps: usize = (self.total_t * 1e3 / self.dt) as usize;
      let mut a = Array2::<f64>::zeros((n_timesteps, self.num_stn));
      for n in 0..self.num_gpe {
        for it in 0..n_timesteps {
          a[[it, n]] = i_ext_py.call1(py, (it as f64 * self.dt, n)).unwrap().extract(py).unwrap();
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

#[pymethods]
impl RubinTerman {
  #[new]
  #[pyo3(signature = (dt = 0.01, total_t = 2.0, parameters_file = PathBuf::from("src/PARAMETERS.toml"),  
      parameters_settings="default".to_owned(), num_stn=10, num_gpe=10, 
      i_ext_stn=default_i_ext_py(), i_ext_gpe=default_i_ext_py(), i_app_gpe=default_i_ext_py()))]
  fn new(
    dt: f64,
    total_t: f64,
    parameters_file: PathBuf,
    parameters_settings: String,
    num_stn: usize,
    num_gpe: usize,
    i_ext_stn: PyObject,
    i_ext_gpe: PyObject,
    i_app_gpe: PyObject,
  ) -> Self {
    Self {
      dt,
      total_t,
      parameters_file,
      parameters_settings,
      num_stn,
      num_gpe,
      i_ext_stn_py: i_ext_stn,
      i_ext_gpe_py: i_ext_gpe,
      i_app_gpe_py: i_app_gpe,
    }
  }

  fn run(&self, py: Python) -> Py<PyDict> {
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
  fn test_vectorize_i_ext() {
    pyo3::prepare_freethreaded_python();
    let rt = RubinTerman {
      dt: 0.01,
      total_t: 1.,
      i_ext_stn_py: Python::with_gil(|py| {
        py.eval(c_str!("lambda t, n: 6.9 if t < 500 else 9.6"), None, None).unwrap().into()
      }),
      ..Default::default()
    };
    let a = rt.vectorize_i_ext(&rt.i_ext_stn_py);
    assert_eq!(a[[0, 0]], 6.9);
    assert_eq!(a[[a.shape()[0] - 1, 0]], 9.6);

    let rt = RubinTerman {
      dt: 0.1,
      total_t: 1.,
      i_ext_stn_py: Python::with_gil(|py| {
        py.eval(c_str!("lambda t, n: 6.9 if t < 500 else 9.6"), None, None).unwrap().into()
      }),
      ..Default::default()
    };
    let a = rt.vectorize_i_ext(&rt.i_ext_stn_py);
    assert_eq!(a[[0, 0]], 6.9);
    assert_eq!(a[[a.shape()[0] - 1, 0]], 9.6);
  }
}
