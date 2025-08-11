use serde::{Deserialize, Serialize};
use struct_field_names_as_array::FieldNamesAsSlice;

use crate::ctx::CTX;
use crate::gpe::GPe;
use crate::gpi::GPi;
use crate::stn::STN;
use crate::str::STR;
use crate::types::*;

#[derive(Deserialize, Serialize, Debug, FieldNamesAsSlice, Clone, Default)]
#[allow(unused)]
pub struct CTXParameters {
  pub base_rate: f64,
  pub stimulated_rate: f64,
  pub max_rate: f64,
  pub sig_s: f64,
  pub min_isi: f64,
  pub syn_rayleigh_sig: f64,
  pub syn_kernel_len: f64,
}

#[derive(Deserialize, Serialize, Debug, FieldNamesAsSlice, Clone, Default)]
#[allow(unused)]
pub struct STRParameters {
  // Conductances
  pub g_l: f64,   // nS/um^2
  pub g_k: f64,   // nS/um^2
  pub g_na: f64,  // nS/um^2
  pub g_t: f64,   // nS/um^2
  pub g_ca: f64,  // nS/um^2
  pub g_ahp: f64, // nS/um^2
  pub g_str: f64, // nS/um^2
  pub g_ctx: f64, // nS/um^2

  // Reversal potentials
  pub v_l: f64,  // mV
  pub v_k: f64,  // mV
  pub v_na: f64, // mV
  pub v_ca: f64, // mV
  pub v_str: f64,
  pub v_ctx: f64,

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

  pub tau_ca: f64,
  pub ca_pre: f64,
  pub ca_post: f64,
}

#[derive(Deserialize, Serialize, Debug, FieldNamesAsSlice, Clone, Default)]
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
  pub g_ctx: f64, // nS/um^2

  // Reversal potentials
  pub v_l: f64,   // mV
  pub v_k: f64,   // mV
  pub v_na: f64,  // mV
  pub v_ca: f64,  // mV
  pub v_g_s: f64, // -85. // mV [MISMATCH]
  pub v_ctx: f64, // -85. // mV [MISMATCH]

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

  // STDP
  pub tau_ca: f64,
  pub ca_pre: f64,
  pub ca_post: f64,
  pub theta_d: f64,
  pub theta_p: f64,
}

#[derive(Serialize, Deserialize, Debug, FieldNamesAsSlice, Clone, Default)]
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
  pub g_str: f64, // nS/um^2

  // Reversal potentials
  pub v_l: f64,   // mV
  pub v_k: f64,   // mV
  pub v_na: f64,  // mV
  pub v_ca: f64,  // mV
  pub v_g_g: f64, // mV
  pub v_s_g: f64, // mV
  pub v_str: f64, // mV

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

  // STDP
  pub tau_ca: f64,
  pub ca_pre: f64,
  pub ca_post: f64,
}

#[derive(Serialize, Deserialize, Debug, FieldNamesAsSlice, Clone, Default)]
#[allow(unused)]
pub struct GPiParameters {
  // Conductances
  pub g_l: f64,   // nS/um^2
  pub g_k: f64,   // nS/um^2
  pub g_na: f64,  // nS/um^2
  pub g_t: f64,   // nS/um^2
  pub g_ca: f64,  // nS/um^2
  pub g_ahp: f64, // nS/um^2
  pub g_s_g: f64, // nS/um^2
  pub g_g_g: f64, // nS/um^2
  pub g_str: f64, // nS/um^2

  // Reversal potentials
  pub v_l: f64,   // mV
  pub v_k: f64,   // mV
  pub v_na: f64,  // mV
  pub v_ca: f64,  // mV
  pub v_g_g: f64, // mV
  pub v_s_g: f64, // mV
  pub v_str: f64, // mV

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

  // STDP
  pub tau_ca: f64,
  pub ca_pre: f64,
  pub ca_post: f64,
}

impl Build<STR, Parameters> for STRParameters {}
impl Build<STN, Parameters> for STNParameters {}
impl Build<GPe, Parameters> for GPeParameters {}
impl Build<GPi, Parameters> for GPiParameters {}
impl Build<CTX, Parameters> for CTXParameters {}

impl PostInit for CTXParameters {
  fn post_init(self) -> Self {
    Self { syn_kernel_len: f64::sqrt(-2. * f64::ln(1. - 0.95)) * self.syn_rayleigh_sig, ..self }
  }
}
impl PostInit for GPeParameters {}
impl PostInit for GPiParameters {}
impl PostInit for STRParameters {}
impl PostInit for STNParameters {
  fn post_init(self) -> Self {
    Self { b_const: 1. / (1. + f64::exp(-self.tht_b / self.sig_b)), ..self }
  }
}

pub type BuilderSTRParameters = Builder<STR, Parameters, STRParameters>;
pub type BuilderSTNParameters = Builder<STN, Parameters, STNParameters>;
pub type BuilderGPeParameters = Builder<GPe, Parameters, GPeParameters>;
pub type BuilderGPiParameters = Builder<GPi, Parameters, GPiParameters>;
pub type BuilderCTXParameters = Builder<CTX, Parameters, CTXParameters>;

//#[cfg(test)]
//mod parameters {
//  use super::*;
//
//  #[test]
//  fn test_default() {
//    let stn = STNParameters::build(true, None, None);
//    assert_eq!(stn.g_l, 2.25);
//    assert_eq!(stn.g_k, 45.);
//    assert_ne!(stn.b_const, f64::NAN);
//    let gpe = GPeParameters::build(true, None, None);
//    assert_eq!(gpe.g_l, 0.1);
//    assert_eq!(gpe.g_k, 30.);
//  }
//
//  #[test]
//  fn test_load_experiment() {
//    let stn = STNParameters::build(true, Some(("wave", None)), None);
//    assert_eq!(stn.g_l, 2.25);
//    assert_eq!(stn.g_g_s, 1.);
//    // TODO: This test caught an interesting thing, when we save an experiment run we dont really save
//    // different version of it... we might be wise to do that =), but seems unnceessary at the
//    // moment
//    //let gpe = GPeParameters::build(true, Some(("wave", Some("v1"))), None);
//    //assert_eq!(gpe.g_g_g, 0.025);
//    //assert_eq!(gpe.g_s_g, 0.03);
//  }
//
//  #[test]
//  fn test_load_custom_experiment() {
//    let gpe = GPeParameters::build(true, Some(("wave", None)), Some(read_map_from_toml("src/TEST.toml", None, "GPe")));
//    assert_eq!(gpe.g_l, 0.1);
//    let stn = STNParameters::build(true, Some(("wave", None)), Some(read_map_from_toml("src/TEST.toml", None, "STN")));
//    assert_eq!(stn.g_l, -99.);
//    assert_eq!(stn.g_g_s, 1.);
//  }
//}
