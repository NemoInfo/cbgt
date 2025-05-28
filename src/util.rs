use ndarray::{Array, Array2, ArrayView, Dimension};
use pyo3::prelude::*;

use std::{
  collections::HashMap,
  io::Write,
  path::Path,
  sync::atomic::{AtomicUsize, Ordering},
};

use crate::{
  gpe::GPePopulationBoundryConditions, stn::STNPopulationBoundryConditions, types::Parameters, Boundary, GPeParameters,
  NeuronData, STNParameters, ToToml, PYF_FILE_NAME, TMP_PYF_FILE_NAME, TMP_PYF_FILE_PATH,
};

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

  pub fn sym_sync_call<F: FnMut()>(&self, mut f: F) {
    self.wait();
    f();
    self.wait();
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

pub fn or_toml_map(base: &mut toml::value::Table, update: toml::value::Table) {
  for (key, val) in update {
    base.entry(key).or_insert(val);
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

pub fn toml_py_function_qualname_to_py_object(val: &toml::Value) -> pyo3::PyObject {
  let qualname = val.as_str().expect("Boundry condition must be stringified python function").to_owned();
  log::debug!("{qualname}");
  let (path_module, fname) = qualname.rsplit_once(".").unwrap();
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
  let num_timesteps: usize = (total_t / dt) as usize;
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
  let num_timesteps: usize = (total_t / dt) as usize;
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
    Some({
      let source = PyModule::import(py, "inspect").ok()?.getattr("getsource").ok()?.call1((f,)).ok()?;

      PyModule::import(py, "textwrap")
        .ok()?
        .getattr("dedent")
        .ok()?
        .call1((source,))
        .ok()?
        .downcast::<pyo3::types::PyString>()
        .ok()?
        .to_string()
    })
  })
}

pub fn get_py_function_source_and_name(f: &PyObject) -> Option<(String, String)> {
  let src = get_py_function_source(&f)?;
  let mut name = get_py_object_name(&f)?;
  if name == "<lambda>" {
    name = src.split_once('=').unwrap().0.trim().to_owned();
  }

  Some((src, name))
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

pub fn array2_to_polars_column(name: &str, array: ndarray::ArrayView2<f64>) -> polars::prelude::Column {
  use polars::prelude::*;
  if array.ncols() == 1 {
    return Float64Chunked::from_vec(name.into(), array.column(0).to_vec()).into_column();
  }

  let mut chunked_builder =
    ListPrimitiveChunkedBuilder::<Float64Type>::new(name.into(), array.nrows(), array.ncols(), DataType::Float64);
  for row in array.axis_iter(ndarray::Axis(0)) {
    match row.as_slice() {
      Some(row) => chunked_builder.append_slice(row),
      None => chunked_builder.append_slice(&row.to_vec()),
    }
  }
  chunked_builder.finish().into_column()
}

pub fn unit_to_polars_column(name: &str, first: ndarray::ArrayView2<f64>, row_count: usize) -> polars::prelude::Column {
  // Que?
  use polars::prelude::*;

  let mut chunked_builder =
    ListPrimitiveChunkedBuilder::<Float64Type>::new(name.into(), row_count, first.len(), DataType::Float64);

  chunked_builder.append_slice(first.as_slice().expect("We can slice an array2"));
  for _ in 1..row_count {
    chunked_builder.append_null();
  }
  chunked_builder.finish().into_column()
}

pub fn strip_uuid_suffix(s: &str) -> String {
  let re = regex::Regex::new(r"_uuid_.*$").unwrap();
  re.replace(s, "").into_owned()
}

pub fn add_uuid_suffix(s: &str) -> String {
  let uuid = uuid::Uuid::new_v4();
  format!("{}_uuid_{}", s, uuid).replace("-", "_")
}

pub fn write_temp_pyf_file(pyf_src: HashMap<String, String>) -> String {
  let file_path = format!("{TMP_PYF_FILE_PATH}/{TMP_PYF_FILE_NAME}.py");
  std::fs::File::create(&file_path).unwrap();
  let mut file = std::fs::OpenOptions::new().append(true).open(&file_path).unwrap();
  log::info!("Saving functions  at [{:?}]", file_path);

  for src in pyf_src.values() {
    writeln!(file, "{src}").unwrap();
  }
  file.flush().unwrap();

  file_path
}

pub fn write_parameter_file(stn: &STNParameters, gpe: &GPeParameters, dir: &str) {
  let parameter_map =
    toml::value::Table::from_iter([("STN".to_owned(), stn.to_toml()), ("GPe".to_owned(), gpe.to_toml())]);

  let file_path: std::path::PathBuf = format!("{dir}/{}", Parameters::EXP_FILE).into();
  let mut file = std::fs::File::create(&file_path).unwrap();
  write!(file, "{}", toml::Value::Table(parameter_map)).unwrap();
  log::info!("Saved parameters at [{}]", file_path.canonicalize().unwrap().display());
}

pub fn write_boundary_file(
  stn: &STNPopulationBoundryConditions,
  gpe: &GPePopulationBoundryConditions,
  dir: &str,
  stn_qual_names: &Vec<String>,
  gpe_qual_names: &Vec<String>,
) {
  let stn_qual_names = stn_qual_names
    .iter()
    .map(|x| format!("{dir}/{PYF_FILE_NAME}.{}", x.rsplit_once(".").unwrap().1))
    .collect::<Vec<_>>();
  let [stn_i_ext_qual_name] = stn_qual_names.as_slice() else {
    panic!("Did not get expected number of qualified python functions");
  };

  let gpe_qual_names = gpe_qual_names
    .iter()
    .map(|x| format!("{dir}/{PYF_FILE_NAME}.{}", x.rsplit_once(".").unwrap().1))
    .collect::<Vec<_>>();
  let [gpe_i_ext_qual_name, gpe_i_app_qual_name] = gpe_qual_names.as_slice() else {
    panic!("Did not get expected number of qualified python functions");
  };

  let bc_map = toml::value::Table::from_iter([
    ("STN".to_owned(), stn.to_toml(&stn_i_ext_qual_name)),
    ("GPe".to_owned(), gpe.to_toml(&gpe_i_ext_qual_name, &gpe_i_app_qual_name)),
  ]);

  let file_path: std::path::PathBuf = format!("{dir}/{}", Boundary::EXP_FILE).into();
  let mut file = std::fs::File::create(&file_path).unwrap();
  write!(file, "{}", toml::Value::Table(bc_map)).unwrap();
  log::info!("Saved parameters at [{}]", file_path.canonicalize().unwrap().display());
}

pub trait InsertAxis<S: ndarray::Data, D: Dimension> {
  fn iax(self, axis: usize) -> ndarray::ArrayBase<S, D::Larger>;
}

impl<S, D> InsertAxis<S, D> for ndarray::ArrayBase<S, D>
where
  S: ndarray::Data,
  D: Dimension,
{
  fn iax(self, axis: usize) -> ndarray::ArrayBase<S, D::Larger> {
    self.insert_axis(ndarray::Axis(axis))
  }
}

pub trait BoolAsf64 {
  fn as_f64(self) -> f64;
}

impl BoolAsf64 for bool {
  fn as_f64(self) -> f64 {
    self as u8 as f64
  }
}
