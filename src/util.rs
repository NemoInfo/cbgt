use ndarray::{Array, Array2, ArrayView, Dimension};

use pyo3::prelude::*;

use std::{
  path::Path,
  sync::atomic::{AtomicUsize, Ordering},
};

use crate::{DEFAULT_PATH, EXPERIMENTS_PATH};

pub fn x_inf<D: Dimension>(v: &ArrayView<f64, D>, tht_x: f64, sig_x: f64) -> Array<f64, D> {
  1. / (1. + ((tht_x - v) / sig_x).exp())
}

pub fn tau_x<D: Dimension>(
  v: &ArrayView<f64, D>,
  tau_x_0: f64,
  tau_x_1: f64,
  tht_x_t: f64,
  sig_x_t: f64,
) -> Array<f64, D> {
  tau_x_0 + tau_x_1 / (1. + ((tht_x_t - v) / sig_x_t).exp())
}

pub struct SpinBarrier {
  count: AtomicUsize,
  generation: AtomicUsize,
  total: usize,
}

impl SpinBarrier {
  pub fn new(total: usize) -> Self {
    Self { count: AtomicUsize::new(total), generation: AtomicUsize::new(0), total }
  }

  pub fn wait(&self) {
    // Record the current generation.
    let gen = self.generation.load(Ordering::Relaxed);
    // Decrement the counter.
    if self.count.fetch_sub(1, Ordering::AcqRel) == 1 {
      // Last thread to arrive.
      self.count.store(self.total, Ordering::Release);
      // Advance to the next generation.
      self.generation.fetch_add(1, Ordering::Release);
    } else {
      // Wait until the generation changes.
      while self.generation.load(Ordering::Acquire) == gen {
        std::hint::spin_loop();
      }
    }
  }
}

pub fn format_number(n: usize) -> String {
  match n {
    0..1_000 => n.to_string(),
    1_000..1_000_000 => format!("{:.1}", n as f32 / 1_000.).trim_end_matches(".0").to_owned() + "K",
    1_000_000..1_000_000_000 => format!("{:.1}", n as f32 / 1_000_000.).trim_end_matches(".0").to_owned() + "M",
    _ => format!("{:.1}", n as f32 / 1_000_000_000.).trim_end_matches(".0").to_owned() + "B",
  }
}

pub trait TryConvertTomlUnit: Sized {
  fn try_convert(val: &toml::Value) -> Result<Self, ()>;
}

impl TryConvertTomlUnit for f64 {
  fn try_convert(val: &toml::Value) -> Result<Self, ()> {
    if let toml::Value::Float(f) = val {
      Ok(*f)
    } else {
      Err(())
    }
  }
}

impl TryConvertTomlUnit for bool {
  fn try_convert(val: &toml::Value) -> Result<Self, ()> {
    if let toml::Value::Boolean(b) = val {
      Ok(*b)
    } else {
      Err(())
    }
  }
}

pub fn try_toml_value_to_1darray<T: TryConvertTomlUnit>(value: &toml::Value) -> Option<ndarray::Array1<T>> {
  let Some(array) = value.as_array() else {
    return None;
  };
  let array = array.iter().map(|x| T::try_convert(x)).collect::<Result<Vec<_>, _>>().ok()?;
  Some(ndarray::Array1::from_vec(array))
}

pub fn try_toml_value_to_2darray<T: TryConvertTomlUnit + Clone>(value: &toml::Value) -> Option<ndarray::Array2<T>> {
  let Some(array) = value.as_array() else {
    return None;
  };
  let arrays = array.iter().map(|x| try_toml_value_to_1darray(x).ok_or(())).collect::<Result<Vec<_>, _>>().ok()?;
  Some(ndarray::stack(ndarray::Axis(0), &arrays.iter().map(|x| x.view()).collect::<Vec<_>>()[..]).unwrap())
}

pub fn update_toml_map(base: &mut toml::value::Table, update: toml::value::Table) {
  for (key, val) in update {
    base.insert(key, val);
  }
}

pub fn read_map_from_toml<P: AsRef<Path>>(file_path: P, version: Option<&str>, map_name: &str) -> toml::value::Table {
  let content =
    std::fs::read_to_string(&file_path).expect(&format!("Failed to read the [{}]:\n", file_path.as_ref().display()));
  let value: toml::Value = content.parse().expect("Failed to parse TOML");
  let mut table = value.as_table().expect("Expected a TOML table at the top level");

  if let Some(version) = version {
    table = table
      .get(version)
      .expect(&format!("Expected [{version}] table at top level"))
      .as_table()
      .expect(&format!("[{version} is not a table]"));
  }

  table
    .get(map_name)
    .unwrap_or(&toml::Value::Table(toml::map::Map::new()))
    .as_table()
    .expect(&format!("[{}.{}] is not a table in {}", version.unwrap_or(""), map_name, file_path.as_ref().display()))
    .to_owned()
}

pub fn build(
  use_default: bool,
  experiment: Option<(&str, Option<&str>)>,
  custom_map: Option<toml::map::Map<String, toml::Value>>,
  map_name: &str,
) -> toml::map::Map<String, toml::Value> {
  let mut map = toml::map::Map::new();

  if use_default {
    update_toml_map(&mut map, read_map_from_toml(DEFAULT_PATH, None, map_name));
  }

  if let Some((experiment, version)) = experiment {
    update_toml_map(&mut map, read_map_from_toml(format!("{EXPERIMENTS_PATH}/{experiment}"), version, map_name));
  }
  if let Some(custom_map) = custom_map {
    update_toml_map(&mut map, custom_map);
  }

  map
}

pub fn py_function_toml_string_to_py_object(val: &toml::Value) -> pyo3::PyObject {
  let qualname = val.as_str().expect("Boundry condition must be stringified python function").to_owned();
  let (path_module, fname) = qualname.split_once(".").unwrap();
  pyo3::prepare_freethreaded_python();
  pyo3::Python::with_gil(|py| {
    let importlib = pyo3::types::PyModule::import(py, "importlib.util").unwrap();

    let spec = importlib.call_method1("spec_from_file_location", ("module.name", format!("{path_module}.py"))).unwrap();
    let module = importlib.call_method1("module_from_spec", (&spec,)).unwrap();

    let loader = spec.getattr("loader").unwrap();
    loader.call_method1("exec_module", (&module,)).expect(
      "Corrupt functions module!\
           Make sure functions are not lambdas passed inline in the constructor",
    );

    module.getattr(fname).expect(&format!("Could not find function {fname}")).into()
  })
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

#[allow(unused)]
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

pub fn get_py_function_source(f: &PyObject) -> Option<String> {
  pyo3::prepare_freethreaded_python();
  Python::with_gil(|py| {
    Some(
      PyModule::import(py, "inspect")
        .ok()?
        .getattr("getsource")
        .ok()?
        .call1((f,))
        .ok()?
        .downcast::<pyo3::types::PyString>()
        .ok()?
        .to_string(),
    )
  })
}

pub fn get_py_object_name(obj: &PyObject) -> Option<String> {
  pyo3::prepare_freethreaded_python();
  Python::with_gil(|py| {
    Some(obj.getattr(py, "__name__").ok()?.downcast_bound::<pyo3::types::PyString>(py).ok()?.to_string())
  })
}

pub fn parse_toml_value(k: &str, v: &str) -> toml::Value {
  let kv = format!("{k}={v}");
  match kv.parse() {
    Ok(v) => v,
    Err(_) => {
      let kv = format!(
        "{k}={}",
        v.replace("array(", "") // numpy.ndarray to toml array
          .replace(")", "")
          .replace(".,", ".0,") // "2." is not valid toml float but "2.0" is
          .replace(".]", ".0]")
          .replace("True", "true") // Python bool to toml bool
          .replace("False", "false")
      );
      kv.parse().expect("Expected numpy array in the form \"array([1,2,3])\"")
    }
  }
}
