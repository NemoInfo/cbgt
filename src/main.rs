use std::path::PathBuf;

use ndarray::{array, s, Array, ArrayView, Dim, Dimension};

mod parameters;
use ndarray_npy::write_npy;
use parameters::*;

/// Rubin Terman model using Euler's method
#[allow(unused)]
struct RubinTerman {
  /// Time between Euler steps (ms)
  dt: f64,
  /// Simulation time (s)
  total_t: f64,
  /// Parameter file path
  parameters_file: PathBuf,
  /// Parameter file path
  parameters_settings: &'static str,
  /// Number of neurons in STN
  num_stn: usize,
  /// Number of neurons in GPe
  num_gpe: usize,
}

impl Default for RubinTerman {
  fn default() -> Self {
    RubinTerman {
      dt: 0.01,
      total_t: 2.0,
      parameters_file: "src/PARAMETERS.toml".into(),
      parameters_settings: "default",
      num_stn: 10,
      num_gpe: 10,
    }
  }
}

impl RubinTerman {
  fn run(&self) -> Array<f64, Dim<[usize; 3]>> {
    let n_timesteps: usize = (self.total_t * 1e3 / self.dt) as usize;
    let stn = STNParameters::from_config(&self.parameters_file, self.parameters_settings);
    let _gpe = GPeParameters::from_config(&self.parameters_file, self.parameters_settings);

    // Create STN state
    let mut v_stn = Array::<f64, _>::zeros((n_timesteps, self.num_stn));
    let mut n_stn = Array::<f64, _>::zeros((n_timesteps, self.num_stn));
    let mut h_stn = Array::<f64, _>::zeros((n_timesteps, self.num_stn));
    let mut r_stn = Array::<f64, _>::zeros((n_timesteps, self.num_stn));
    let mut ca_stn = Array::<f64, _>::zeros((n_timesteps, self.num_stn));

    // Create STN currents
    let mut i_l_stn = Array::<f64, _>::zeros((n_timesteps, self.num_stn));
    let mut i_k_stn = Array::<f64, _>::zeros((n_timesteps, self.num_stn));
    let mut i_na_stn = Array::<f64, _>::zeros((n_timesteps, self.num_stn));
    let mut i_t_stn = Array::<f64, _>::zeros((n_timesteps, self.num_stn));
    let mut i_ca_stn = Array::<f64, _>::zeros((n_timesteps, self.num_stn));
    let mut i_ahp_stn = Array::<f64, _>::zeros((n_timesteps, self.num_stn));

    v_stn.slice_mut(s![0, ..]).assign(&array![
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
    h_stn.slice_mut(s![0, ..]).assign(&array![
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
    n_stn.slice_mut(s![0, ..]).assign(&array![
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
    r_stn.slice_mut(s![0, ..]).assign(&array![
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
    ca_stn.slice_mut(s![0, ..]).assign(&array![
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

    for it in 0..n_timesteps - 1 {
      let t = s![it, ..];
      let t1 = s![it + 1, ..];

      let ((v, mut v1), (r, mut r1), (n, mut n1), (h, mut h1), (ca, mut ca1)) = (
        v_stn.multi_slice_mut((t, t1)),
        r_stn.multi_slice_mut((t, t1)),
        n_stn.multi_slice_mut((t, t1)),
        h_stn.multi_slice_mut((t, t1)),
        ca_stn.multi_slice_mut((t, t1)),
      );
      let (v, r, n, h, ca) = (&v.view(), &r.view(), &n.view(), &h.view(), &ca.view());

      let n_inf = &x_inf(v, stn.tht_n, stn.sig_n);
      let m_inf = &x_inf(v, stn.tht_m, stn.sig_m);
      let h_inf = &x_inf(v, stn.tht_h, stn.sig_h);
      let a_inf = &x_inf(v, stn.tht_a, stn.sig_a);
      let r_inf = &x_inf(v, stn.tht_r, stn.sig_r);
      let s_inf = &x_inf(v, stn.tht_s, stn.sig_s);
      let b_inf = &x_inf(r, stn.tht_b, -stn.sig_b) - stn.b_const; // [!]

      let tau_n = &tau_x(v, stn.tau_n_0, stn.tau_n_1, stn.tht_n_t, stn.sig_n_t);
      let tau_h = &tau_x(v, stn.tau_h_0, stn.tau_h_1, stn.tht_h_t, stn.sig_h_t);
      let tau_r = &tau_x(v, stn.tau_r_0, stn.tau_r_1, stn.tht_r_t, stn.sig_r_t);

      // Compute currents
      let mut i_l_stn_mut = i_l_stn.slice_mut(t);
      let mut i_k_stn_mut = i_k_stn.slice_mut(t);
      let mut i_na_stn_mut = i_na_stn.slice_mut(t);
      let mut i_t_stn_mut = i_t_stn.slice_mut(t);
      let mut i_ca_stn_mut = i_ca_stn.slice_mut(t);
      let mut i_ahp_stn_mut = i_ahp_stn.slice_mut(t);

      i_l_stn_mut.assign(&(stn.g_l * (v - stn.v_l)));
      i_k_stn_mut.assign(&(stn.g_k * n.powi(4) * (v - stn.v_k)));
      i_na_stn_mut.assign(&(stn.g_na * m_inf.powi(3) * h * (v - stn.v_na)));
      i_t_stn_mut.assign(&(stn.g_t * a_inf.powi(3) * b_inf.pow2() * (v - stn.v_ca)));
      i_ca_stn_mut.assign(&(stn.g_ca * s_inf.powi(2) * (v - stn.v_ca)));
      i_ahp_stn_mut.assign(&(stn.g_ahp * (v - stn.v_k) * ca / (ca + stn.k_1)));


      // Update state
      v1.assign(
        &(v + self.dt * (-&i_l_stn_mut - &i_k_stn_mut - &i_na_stn_mut - &i_t_stn_mut - &i_ca_stn_mut - &i_ahp_stn_mut)), // - i_ext_stn
      );
      n1.assign(&(n + self.dt * stn.phi_n * (n_inf - n) / tau_n));
      h1.assign(&(h + self.dt * stn.phi_h * (h_inf - h) / tau_h));
      r1.assign(&(r + self.dt * stn.phi_r * (r_inf - r) / tau_r));
      ca1.assign(&(ca + self.dt * stn.eps * ((-&i_ca_stn_mut - &i_t_stn_mut) - stn.k_ca * ca)));
    }

    let mut combined = Array::<f64, _>::zeros((7, n_timesteps, self.num_stn));
    combined.slice_mut(s![0, .., ..]).assign(&v_stn);
    combined.slice_mut(s![1, .., ..]).assign(&i_l_stn);
    combined.slice_mut(s![2, .., ..]).assign(&i_k_stn);
    combined.slice_mut(s![3, .., ..]).assign(&i_na_stn);
    combined.slice_mut(s![4, .., ..]).assign(&i_t_stn);
    combined.slice_mut(s![5, .., ..]).assign(&i_ca_stn);
    combined.slice_mut(s![6, .., ..]).assign(&i_ahp_stn);

    combined
  }
}

fn x_inf<D: Dimension>(v: &ArrayView<f64, D>, tht_x: f64, sig_x: f64) -> Array<f64, D> {
  1. / (1. + ((tht_x - v) / sig_x).exp())
}

fn tau_x<D: Dimension>(v: &ArrayView<f64, D>, tau_x_0: f64, tau_x_1: f64, tht_x_t: f64, sig_x_t: f64) -> Array<f64, D> {
  tau_x_0 + tau_x_1 / (1. + ((tht_x_t - v) / sig_x_t).exp())
}

fn main() {
  let res = RubinTerman {
    dt: 0.01,
    total_t: 2.,
    ..Default::default()
  }
  .run();
  println!("Simulation completed!");

  let file_name = "test.npy";
  write_npy(&file_name, &res).expect(&format!("Failed to write results to {file_name}"));
  println!("Wrote results to {file_name}!");
}
