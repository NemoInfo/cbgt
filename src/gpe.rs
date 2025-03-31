use ndarray::{s, Array1, Array2, ArrayView1};
use toml::map::Map;
use toml::Value;

use std::collections::HashMap;
use std::path::Path;

use crate::parameters::GPeParameters;
use crate::util::*;

pub struct GPePopulation {
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
  pub i_app: Array2<f64>,

  // Connection Matrices
  pub c_g_g: Array2<f64>,
  pub c_s_g: Array2<f64>,

  // Connection Currents
  pub i_g_g: Array2<f64>,
  pub i_s_g: Array2<f64>,
}

impl GPePopulation {
  pub fn set_ics_from_config<P: AsRef<Path>>(&mut self, file_path: P, version: &str) {
    let content = std::fs::read_to_string(file_path).expect("Failed to read the config file");
    let value: Value = content.parse().expect("Failed to parse TOML");
    let table = value.as_table().expect("Expected a TOML table at the top level");

    let pop_type = "GPe";
    let type_table = table
      .get(pop_type)
      .expect(&format!("Missing [{}] section", pop_type))
      .as_table()
      .expect(&format!("[{}] is not a table", pop_type));

    let default_table = type_table
      .get("default")
      .expect(&format!("missing [{}.default]", pop_type))
      .as_table()
      .expect(&format!("[{}.default] is not a table", pop_type));

    let Some(default_ics) = default_table.get("init") else {
      println!("[WARN] [GPe.default.init] not found, setting all inner state to 0");
      return;
    };
    let default_ics = default_ics.as_table().expect("[GPe.default.init] should be a table");
    self.load_ics_from_table(&default_ics);

    if version == "default" {
      return;
    }

    let alt_table = type_table
      .get(version)
      .expect(&format!("missing [{}.{version}]", pop_type))
      .as_table()
      .expect(&format!("[{}.{version}] is not a table", pop_type));

    if let Some(alt_ics) = alt_table.get("init") {
      let alt_ics = alt_ics.as_table().expect(&format!("[GPe.{version}.init] should be a table"));
      self.load_ics_from_table(&alt_ics);
    }
  }
  fn load_ics_from_table(&mut self, ic_table: &Map<String, Value>) {
    let mut state = HashMap::from([
      ("v", &mut self.v),
      ("n", &mut self.n),
      ("h", &mut self.h),
      ("r", &mut self.r),
      ("ca", &mut self.ca),
      ("s", &mut self.s),
    ]);
    for (k, v) in ic_table {
      if let Some(state) = state.get_mut(k.as_str()) {
        if state.shape()[1] != v.as_array().expect("array").len() {
          return;
        }
        state.row_mut(0).assign(&Array1::<f64>::from(
          v.as_array()
            .expect("Initial condition entry should be array")
            .iter()
            .map(|x| {
              x.as_float()
                .or(x.as_integer().and_then(|x| Some(x as f64)))
                .expect(&format!("Expected integer or floating point, got {x:?}"))
            })
            .collect::<Vec<f64>>(),
        ));
      } else {
        println!("[WARN] initial condition entry {k} is invalid");
      }
    }
  }

  pub fn new(
    num_timesteps: usize,
    num_neurons: usize,
    i_ext: Array2<f64>,
    i_app: Array2<f64>,
    c_s_g: Array2<f64>,
    c_g_g: Array2<f64>,
  ) -> Self {
    GPePopulation {
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
      i_s_g: Array2::zeros((num_timesteps, num_neurons)),
      i_g_g: Array2::zeros((num_timesteps, num_neurons)),
      i_ext,
      i_app,
      c_s_g,
      c_g_g,
    }
  }

  pub fn euler_step(&mut self, it: usize, dt: f64, gpe: &GPeParameters, s_stn: &ArrayView1<f64>) {
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

    let n_inf = &x_inf(v, gpe.tht_n, gpe.sig_n);
    let m_inf = &x_inf(v, gpe.tht_m, gpe.sig_m);
    let h_inf = &x_inf(v, gpe.tht_h, gpe.sig_h);
    let a_inf = &x_inf(v, gpe.tht_a, gpe.sig_a);
    let r_inf = &x_inf(v, gpe.tht_r, gpe.sig_r);
    let s_inf = &x_inf(v, gpe.tht_s, gpe.sig_s);

    let tau_n = &tau_x(v, gpe.tau_n_0, gpe.tau_n_1, gpe.tht_n_t, gpe.sig_n_t);
    let tau_h = &tau_x(v, gpe.tau_h_0, gpe.tau_h_1, gpe.tht_h_t, gpe.sig_h_t);

    // Compute currents
    let mut i_l = self.i_l.slice_mut(t);
    let mut i_k = self.i_k.slice_mut(t);
    let mut i_na = self.i_na.slice_mut(t);
    let mut i_t = self.i_t.slice_mut(t);
    let mut i_ca = self.i_ca.slice_mut(t);
    let mut i_ahp = self.i_ahp.slice_mut(t);
    let mut i_s_g = self.i_s_g.slice_mut(t);
    let mut i_g_g = self.i_g_g.slice_mut(t);

    i_l.assign(&(gpe.g_l * (v - gpe.v_l)));
    i_k.assign(&(gpe.g_k * n.powi(4) * (v - gpe.v_k)));
    i_na.assign(&(gpe.g_na * m_inf.powi(3) * h * (v - gpe.v_na)));
    i_t.assign(&(gpe.g_t * a_inf.powi(3) * r * (v - gpe.v_ca)));
    i_ca.assign(&(gpe.g_ca * s_inf.powi(2) * (v - gpe.v_ca)));
    i_ahp.assign(&(gpe.g_ahp * (v - gpe.v_k) * ca / (ca + gpe.k_1)));
    i_s_g.assign(&(gpe.g_s_g * (v - gpe.v_s_g) * (self.c_s_g.t().dot(s_stn))));
    i_g_g.assign(&(gpe.g_g_g * (v - gpe.v_g_g) * (self.c_g_g.t().dot(s))));

    // Update state
    v1.assign(
      &(v
        + dt
          * (-&i_l - &i_k - &i_na - &i_t - &i_ca - &i_ahp - &i_s_g - &i_g_g - &self.i_ext.row(it)
            + &self.i_app.row(it))),
    );
    n1.assign(&(n + dt * gpe.phi_n * (n_inf - n) / tau_n));
    h1.assign(&(h + dt * gpe.phi_h * (h_inf - h) / tau_h));
    r1.assign(&(r + dt * gpe.phi_r * (r_inf - r) / gpe.tau_r));
    ca1.assign(&(ca + dt * gpe.eps * ((-&i_ca - &i_t) - gpe.k_ca * ca)));

    // Update synapses
    let h_syn_inf = x_inf(&(v - gpe.tht_g).view(), gpe.tht_g_h, gpe.sig_g_h);
    s1.assign(&(s + dt * (gpe.alpha * h_syn_inf * (1. - s) - gpe.beta * s)));
  }
}
