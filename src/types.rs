use std::collections::HashMap;

use crate::{util::*, PYF_FILE_NAME, TMP_PYF_FILE_NAME, TMP_PYF_FILE_PATH};

pub trait Neuron {
  const TYPE: &'static str;
}

pub trait NeuronData {
  type Default;
  const EXP_FILE: &'static str;
  const DEFAULT: Option<Self::Default>;
}

pub trait Build<N: Neuron, ND: NeuronData> {
  const PYTHON_CALLABLE_FIELD_NAMES: &[&'static str] = &[];
}

pub struct Builder<N, ND, T>
where
  N: Neuron,
  ND: NeuronData,
  T: Build<N, ND>,
{
  pub map: toml::map::Map<String, toml::Value>,
  _mark: std::marker::PhantomData<(N, ND, T)>,
}

impl<N, ND, T> Builder<N, ND, T>
where
  N: Neuron,
  ND: NeuronData,
  T: Build<N, ND>,
  Builder<N, ND, T>: BuildDefault<N, ND, T>,
{
  fn new(map: toml::map::Map<String, toml::Value>) -> Self {
    Self { map, _mark: std::marker::PhantomData }
  }

  fn xor_custom_file(mut self, file_path: Option<&str>) -> Self {
    assert!(file_path.is_none() || self.map.is_empty());
    if let Some(file_path) = file_path {
      self.map = read_map_from_toml(file_path, None, "STN");
    }
    self
  }

  fn or_experiment(mut self, experiment: Option<(&str, Option<&str>)>) -> Self {
    if let Some((experiment, version)) = experiment {
      let base = read_map_from_toml(format!("{}/{experiment}/{}", EXP_PATH, ND::EXP_FILE), version, N::TYPE);
      or_toml_map(&mut self.map, base);
    }
    self
  }

  pub fn build(
    kw_map: toml::map::Map<String, toml::Value>,
    custom_file: Option<&str>,
    experiment: Option<(&str, Option<&str>)>,
    with_default: bool,
  ) -> Self {
    if with_default {
      Self::new(kw_map).xor_custom_file(custom_file).or_experiment(experiment).or_default()
    } else {
      Self::new(kw_map).xor_custom_file(custom_file).or_experiment(experiment)
    }
  }
}

const EXP_PATH: &'static str = "experiments/";

pub enum Parameters {}
impl NeuronData for Parameters {
  type Default = &'static str;
  const DEFAULT: Option<Self::Default> = Some("src/DEFAULT.toml");
  const EXP_FILE: &'static str = "PARAMETERS.toml";
}

pub enum Boundary {}
impl NeuronData for Boundary {
  type Default = (&'static str, &'static str);
  const DEFAULT: Option<Self::Default> = Some(("zeroed", "zeroed = lambda t,n: 0.\n"));
  const EXP_FILE: &'static str = "BOUNDRY.toml";
}

pub trait PostInit: Sized {
  fn post_init(self) -> Self {
    self
  }
}

pub trait BuildDefault<N, ND, T>
where
  N: Neuron,
  ND: NeuronData,
  T: Build<N, ND>,
{
  fn or_default(self) -> Self;
}

impl<N, T> BuildDefault<N, Parameters, T> for Builder<N, Parameters, T>
where
  N: Neuron,
  T: Build<N, Parameters>,
{
  fn or_default(mut self) -> Self {
    if let Some(file_path) = Parameters::DEFAULT {
      let base = read_map_from_toml(file_path, None, N::TYPE);
      or_toml_map(&mut self.map, base);
    }
    self
  }
}

impl<N, T> Builder<N, Parameters, T>
where
  N: Neuron,
  T: Build<N, Parameters> + PostInit + serde::de::DeserializeOwned,
{
  pub fn finish(self) -> T {
    toml::Value::Table(self.map)
      .try_into::<T>()
      .unwrap_or_else(|err| panic!("Failed to deserialize {} parameters:\n{err}", N::TYPE))
      .post_init()
  }
}

impl<N, T> Builder<N, Boundary, T>
where
  N: Neuron,
  T: Build<N, Boundary>,
{
  pub fn get_count(&self) -> Option<usize> {
    self
      .map
      .get("count".into())
      .map(|x| x.as_integer().expect(&format!("{}_count must be usize", N::TYPE.to_lowercase())) as usize)
  }

  pub fn extends_pyf_src(&self, pyf_src: &mut HashMap<String, String>) {
    for &key in T::PYTHON_CALLABLE_FIELD_NAMES {
      let Some(value) = self.map.get(key.into()) else {
        continue;
      };
      let qname = value.as_str().expect(&format!("{}_{key}", N::TYPE.to_lowercase()));
      let (file_path, func_name) = qname.rsplit_once(".").expect("Python function qualified name");
      let (_, file_name) = file_path.rsplit_once("/").expect("Python function qualified name");

      if file_name == TMP_PYF_FILE_NAME {
        continue;
      }
      assert!(file_name == PYF_FILE_NAME);

      let py_object = toml_py_function_qualname_to_py_object(&file_name.into());
      let src = get_py_function_source(&py_object).expect("Could not get source");
      let new_func_name = add_uuid_suffix(&strip_uuid_suffix(func_name));

      let qname = format!("{TMP_PYF_FILE_PATH}/{TMP_PYF_FILE_NAME}.{new_func_name}");
      let src = src.replace(func_name, &new_func_name);

      pyf_src.insert(qname.into(), src);
    }
  }
}

impl<N, T> BuildDefault<N, Boundary, T> for Builder<N, Boundary, T>
where
  N: Neuron,
  T: Build<N, Boundary>,
{
  fn or_default(mut self) -> Self {
    if let Some((qname, _src)) = Boundary::DEFAULT {
      for &key in T::PYTHON_CALLABLE_FIELD_NAMES {
        self.map.entry(key).or_insert(format!("{TMP_PYF_FILE_PATH}/{TMP_PYF_FILE_NAME}.{qname}").into());
      }
    }
    self
  }
}

pub trait ToToml<N: Neuron, ND: NeuronData>: Build<N, ND> {
  fn to_toml(&self) -> toml::Value;
}

impl<N: Neuron, T: Build<N, Parameters> + serde::Serialize> ToToml<N, Parameters> for T {
  fn to_toml(&self) -> toml::Value {
    let table = toml::Value::try_from(&self).unwrap();
    assert!(table.is_table());
    table
  }
}
