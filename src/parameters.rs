use serde::{de::DeserializeOwned, Deserialize, Serialize};
use struct_field_names_as_array::FieldNamesAsArray;
use toml::Value;

use crate::util::*;

#[derive(Deserialize, Serialize, Debug, FieldNamesAsArray, Clone)]
#[allow(unused)]
pub struct STNParameters {
  // Conductances
  pub g_l: f64,   // nS/um^2
  pub g_k: f64,   // nS/um^2
  pub g_na: f64,  // nS/um^2
  pub g_t: f64,   // nS/um^2
  pub g_ca: f64,  // nS/um^2
  pub g_ahp: f64, // nS/um^2
  pub g_g_s: f64, // nS/um^2

  // Reversal potentials
  pub v_l: f64,   // mV
  pub v_k: f64,   // mV
  pub v_na: f64,  // mV
  pub v_ca: f64,  // mV
  pub v_g_s: f64, // -85. // mV [MISMATCH]

  // Time constants
  pub tau_h_1: f64, // ms
  pub tau_n_1: f64, // ms
  pub tau_r_1: f64, // ms
  pub tau_h_0: f64, // ms
  pub tau_n_0: f64, // ms
  pub tau_r_0: f64, // 40. // ms [MISMATCH]

  pub phi_h: f64,
  pub phi_n: f64,
  pub phi_r: f64, // 0.2 // [MISMATCH]

  // Calcium parameters
  pub k_1: f64,
  pub k_ca: f64,
  pub eps: f64, // ms^-1 x (== phi_h * 5e-5)

  // Threshold potentials
  pub tht_m: f64, // mV x
  pub tht_h: f64, // mV x
  pub tht_n: f64, // mV x
  pub tht_r: f64, // mV
  pub tht_a: f64, // mV
  pub tht_b: f64, // 0.4 [MISMATCH]
  pub tht_s: f64, // mV x

  // Tau threshold potentials
  pub tht_h_t: f64, // mV
  pub tht_n_t: f64, // mV
  pub tht_r_t: f64, // mV

  // Synaptic threshold potentials
  pub tht_g_h: f64, // mV
  pub tht_g: f64,   // mV

  // Synaptic rate constants
  pub alpha: f64, // ms^-1
  pub beta: f64,  // ms^-1

  // Sigmoid slopes
  pub sig_m: f64,
  pub sig_h: f64,
  pub sig_n: f64,
  pub sig_r: f64,
  pub sig_a: f64,
  pub sig_b: f64, // -0.1 [MISMATCH]
  pub sig_s: f64,

  // Tau sigmoid slopes
  pub sig_h_t: f64,
  pub sig_n_t: f64,
  pub sig_r_t: f64,

  // Synaptic sigmoid slopes
  pub sig_g_h: f64,

  // Constant parameter
  pub b_const: f64,
}

#[derive(Serialize, Deserialize, Debug, FieldNamesAsArray, Clone)]
#[allow(unused)]
pub struct GPeParameters {
  // Conductances
  pub g_l: f64,   // nS/um^2
  pub g_k: f64,   // nS/um^2
  pub g_na: f64,  // nS/um^2
  pub g_t: f64,   // nS/um^2
  pub g_ca: f64,  // nS/um^2
  pub g_ahp: f64, // nS/um^2
  pub g_s_g: f64, // nS/um^2
  pub g_g_g: f64, // nS/um^2

  // Reversal potentials
  pub v_l: f64,   // mV
  pub v_k: f64,   // mV
  pub v_na: f64,  // mV
  pub v_ca: f64,  // mV
  pub v_g_g: f64, // mV
  pub v_s_g: f64, // mV

  // Time constants
  pub tau_h_1: f64, // ms
  pub tau_n_1: f64, // ms
  pub tau_h_0: f64, // ms
  pub tau_n_0: f64, // ms
  pub tau_r: f64,   // ms

  pub phi_h: f64,
  pub phi_n: f64,
  pub phi_r: f64,

  // Calcium parameters
  pub k_1: f64,
  pub k_ca: f64,
  pub eps: f64, // ms^-1

  // Threshold potentials
  pub tht_m: f64, // mV
  pub tht_h: f64, // mV
  pub tht_n: f64, // mV
  pub tht_r: f64, // mV
  pub tht_a: f64, // mV
  pub tht_s: f64, // mV

  // Tau threshold potentials
  pub tht_h_t: f64, // mV
  pub tht_n_t: f64, // mV

  // Synaptic threshold potentials
  pub tht_g_h: f64, // mV
  pub tht_g: f64,   // mV

  // Synaptic rate constants
  pub alpha: f64, // ms^-1
  pub beta: f64,  // ms^-1

  // Sigmoid slopes
  pub sig_m: f64,
  pub sig_h: f64,
  pub sig_n: f64,
  pub sig_r: f64,
  pub sig_a: f64,
  pub sig_s: f64,

  // Tau sigmoid slopes
  pub sig_h_t: f64,
  pub sig_n_t: f64,

  // Synaptic sigmoid slope
  pub sig_g_h: f64,
}

impl ModelDescription for STNParameters {
  const TYPE: &'static str = "STN";
  const EXPERIMENT_FILE_NAME: &'static str = EXPERIMENT_PARAMETER_FILE_NAME;
}

impl Parameters for STNParameters {
  fn post_init(self) -> Self {
    Self { b_const: 1. / (1. + f64::exp(-self.tht_b / self.sig_b)), ..self }
  }
}

impl ModelDescription for GPeParameters {
  const TYPE: &'static str = "GPe";
  const EXPERIMENT_FILE_NAME: &'static str = EXPERIMENT_PARAMETER_FILE_NAME;
}

impl Parameters for GPeParameters {}

pub const DEFAULT_PATH: &'static str = "src/DEFAULT.toml";
pub const EXPERIMENTS_PATH: &'static str = "experiments";
pub const EXPERIMENT_PARAMETER_FILE_NAME: &'static str = "PARAMETERS.toml";
pub const EXPERIMENT_BC_FILE_NAME: &'static str = "BOUNDRY.toml";

pub trait ModelDescription {
  const TYPE: &'static str;
  const EXPERIMENT_FILE_NAME: &'static str;

  fn build_map(
    use_default: bool,
    experiment: Option<(&str, Option<&str>)>,
    custom_map: Option<toml::value::Table>,
  ) -> toml::value::Table {
    let mut s = String::new();
    let map = build(
      use_default,
      experiment.map(|(p, v)| {
        s = format!("{p}/{EXPERIMENT_PARAMETER_FILE_NAME}");
        (s.as_str(), v)
      }),
      custom_map,
      Self::TYPE,
    );

    map
  }
}

pub trait Parameters: Serialize + DeserializeOwned + ModelDescription {
  fn post_init(self) -> Self {
    self
  }

  fn build(
    use_default: bool,
    experiment: Option<(&str, Option<&str>)>,
    custom_map: Option<toml::value::Table>,
  ) -> Self {
    toml::Value::Table(Self::build_map(use_default, experiment, custom_map))
      .try_into::<Self>()
      .unwrap_or_else(|err| panic!("Failed to deserialize {} parameters:\n{err}", Self::TYPE))
      .post_init()
  }

  fn to_toml(&self) -> toml::Value {
    let table = Value::try_from(&self).unwrap();
    assert!(table.is_table());
    table
  }
}

#[cfg(test)]
mod parameters {
  use super::*;

  #[test]
  fn test_default() {
    let stn = STNParameters::build(true, None, None);
    assert_eq!(stn.g_l, 2.25);
    assert_eq!(stn.g_k, 45.);
    assert_ne!(stn.b_const, f64::NAN);
    let gpe = GPeParameters::build(true, None, None);
    assert_eq!(gpe.g_l, 0.1);
    assert_eq!(gpe.g_k, 30.);
  }

  #[test]
  fn test_load_experiment() {
    let stn = STNParameters::build(true, Some(("wave_rt", None)), None);
    assert_eq!(stn.g_l, 2.25);
    assert_eq!(stn.g_g_s, 1.);
    let gpe = GPeParameters::build(true, Some(("wave_rt", Some("v1"))), None);
    assert_eq!(gpe.g_g_g, 0.025);
    assert_eq!(gpe.g_s_g, 0.03);
  }

  #[test]
  fn test_load_custom_experiment() {
    let gpe =
      GPeParameters::build(true, Some(("wave_rt", None)), Some(read_map_from_toml("src/TEST.toml", None, "GPe")));
    assert_eq!(gpe.g_l, 0.1);
    let stn =
      STNParameters::build(true, Some(("wave_rt", None)), Some(read_map_from_toml("src/TEST.toml", None, "STN")));
    assert_eq!(stn.g_l, -99.);
    assert_eq!(stn.g_g_s, 1.);
  }
}
