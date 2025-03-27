use serde::{de::DeserializeOwned, Deserialize};
use std::{fs, path::Path};
use toml::Value;

#[derive(Deserialize, Debug)]
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

#[derive(Deserialize, Debug)]
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

impl Parameters for STNParameters {
  const TYPE: &'static str = "STN";

  fn update(self) -> Self {
    Self {
      b_const: 1. / (1. + f64::exp(-self.tht_b / self.sig_b)),
      ..self
    }
  }
}
impl Parameters for GPeParameters {
  const TYPE: &'static str = "GPe";
}

pub trait Parameters: DeserializeOwned {
  const TYPE: &'static str;
  fn update(self) -> Self {
    self
  }

  fn from_config<P: AsRef<Path>>(file_path: P, version: &str) -> Self {
    let content = fs::read_to_string(file_path).expect("Failed to read the config file");
    let value: Value = content.parse().expect("Failed to parse TOML");
    let table = value.as_table().expect("Expected a TOML table at the top level");

    let type_table = table
      .get(Self::TYPE)
      .expect(&format!("Missing [{}] section", Self::TYPE))
      .as_table()
      .expect(&format!("[{}] is not a table", Self::TYPE));

    let default_table = type_table
      .get("default")
      .expect(&format!("Missing [{}.default]", Self::TYPE))
      .as_table()
      .expect(&format!("[{}.default] is not a table", Self::TYPE));

    let mut merged = default_table.clone();

    if version == "default" {
      return Value::Table(merged).try_into::<Self>().unwrap_or_else(|err| panic!("Failed to deserialize parameters: {err}")).update();
    }

    type_table
      .get(version)
      .expect(&format!("Missing [{}.{version}]", Self::TYPE))
      .as_table()
      .expect(&format!("[{}.{version}] is not a table", Self::TYPE))
      .iter()
      .for_each(|(key, val)| _ = merged.insert(key.clone(), val.clone()));

    Value::Table(merged).try_into::<Self>().unwrap_or_else(|err| panic!("Failed to deserialize parameters: {err}")).update()
  }
}
