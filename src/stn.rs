use ndarray::{s, Array2};

use crate::parameters::STNParameters;
use crate::util::*;

pub struct STNPopulation {
  // State
  pub v: Array2<f64>,
  pub n: Array2<f64>,
  pub h: Array2<f64>,
  pub r: Array2<f64>,
  pub ca: Array2<f64>,
  pub s: Array2<f64>,

  // Currents
  pub i_l: Array2<f64>,
  pub i_k: Array2<f64>,
  pub i_na: Array2<f64>,
  pub i_t: Array2<f64>,
  pub i_ca: Array2<f64>,
  pub i_ahp: Array2<f64>,
  pub i_ext: Array2<f64>,
}

impl STNPopulation {
  pub fn new(num_timesteps: usize, num_neurons: usize, i_ext: Array2<f64>) -> Self {
    STNPopulation {
      v: Array2::zeros((num_timesteps, num_neurons)),
      n: Array2::zeros((num_timesteps, num_neurons)),
      h: Array2::zeros((num_timesteps, num_neurons)),
      r: Array2::zeros((num_timesteps, num_neurons)),
      ca: Array2::zeros((num_timesteps, num_neurons)),
      s: Array2::zeros((num_timesteps, num_neurons)),
      i_l: Array2::zeros((num_timesteps, num_neurons)),
      i_k: Array2::zeros((num_timesteps, num_neurons)),
      i_na: Array2::zeros((num_timesteps, num_neurons)),
      i_t: Array2::zeros((num_timesteps, num_neurons)),
      i_ca: Array2::zeros((num_timesteps, num_neurons)),
      i_ahp: Array2::zeros((num_timesteps, num_neurons)),
      i_ext,
    }
  }

  pub fn euler_step(&mut self, it: usize, dt: f64, stn: &STNParameters) {
    let t = s![it, ..];
    let t1 = s![it + 1, ..];

    let ((v, mut v1), (r, mut r1), (n, mut n1), (h, mut h1), (ca, mut ca1), (s, mut s1)) = (
      self.v.multi_slice_mut((t, t1)),
      self.r.multi_slice_mut((t, t1)),
      self.n.multi_slice_mut((t, t1)),
      self.h.multi_slice_mut((t, t1)),
      self.ca.multi_slice_mut((t, t1)),
      self.s.multi_slice_mut((t, t1)),
    );
    let (v, r, n, h, ca, s) = (&v.view(), &r.view(), &n.view(), &h.view(), &ca.view(), &s.view());

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
    let mut i_l = self.i_l.slice_mut(t);
    let mut i_k = self.i_k.slice_mut(t);
    let mut i_na = self.i_na.slice_mut(t);
    let mut i_t = self.i_t.slice_mut(t);
    let mut i_ca = self.i_ca.slice_mut(t);
    let mut i_ahp = self.i_ahp.slice_mut(t);

    i_l.assign(&(stn.g_l * (v - stn.v_l)));
    i_k.assign(&(stn.g_k * n.powi(4) * (v - stn.v_k)));
    i_na.assign(&(stn.g_na * m_inf.powi(3) * h * (v - stn.v_na)));
    i_t.assign(&(stn.g_t * a_inf.powi(3) * b_inf.pow2() * (v - stn.v_ca)));
    i_ca.assign(&(stn.g_ca * s_inf.powi(2) * (v - stn.v_ca)));
    i_ahp.assign(&(stn.g_ahp * (v - stn.v_k) * ca / (ca + stn.k_1)));

    // Update state
    v1.assign(&(v + dt * (-&i_l - &i_k - &i_na - &i_t - &i_ca - &i_ahp - &self.i_ext.row(it))));
    n1.assign(&(n + dt * stn.phi_n * (n_inf - n) / tau_n));
    h1.assign(&(h + dt * stn.phi_h * (h_inf - h) / tau_h));
    r1.assign(&(r + dt * stn.phi_r * (r_inf - r) / tau_r));
    ca1.assign(&(ca + dt * stn.eps * ((-&i_ca - &i_t) - stn.k_ca * ca)));

    // Update synapses
    let h_syn_inf = x_inf(&(v - stn.tht_g).view(), stn.tht_g_h, stn.sig_g_h);
    s1.assign(&(s + dt * (stn.alpha * h_syn_inf * (1. - s) - stn.beta * s)));
  }
}
