use ndarray::{s, Array1, Array2, ArrayView1};
use struct_field_names_as_array::FieldNamesAsArray;
use toml::map::Map;
use toml::Value;

use std::collections::HashMap;
use std::path::Path;

use crate::parameters::GPeParameters;
use crate::{util::*, ModelDescription, EXPERIMENT_BC_FILE_NAME};

#[derive(FieldNamesAsArray)]
pub struct GPePopulationBoundryConditions {
  pub count: usize,
  // State
  pub v: Array1<f64>,
  pub n: Array1<f64>,
  pub h: Array1<f64>,
  pub r: Array1<f64>,
  pub ca: Array1<f64>,
  pub s: Array1<f64>,
  pub i_ext: Array2<f64>,
  pub i_app: Array2<f64>,

  // Connection Matrice
  pub c_g_g: Array2<f64>,
  pub c_s_g: Array2<f64>,
}

impl ModelDescription for GPePopulationBoundryConditions {
  const TYPE: &'static str = "GPe";
  const EXPERIMENT_FILE_NAME: &'static str = EXPERIMENT_BC_FILE_NAME;
}

impl GPePopulationBoundryConditions {
  pub fn from(
    map: toml::map::Map<String, toml::Value>,
    gpe_count: usize,
    stn_count: usize,
    dt: f64,
    total_t: f64,
  ) -> Self {
    let num_timesteps: usize = (total_t * 1e3 / dt) as usize;

    let pbc = Array1::zeros(gpe_count);

    let v = map.get("v").map(try_toml_value_to_1darray::<f64>).map_or(pbc.clone(), |x| x.expect("invalid bc dim v"));
    assert_eq!(v.len(), gpe_count);
    let n = map.get("n").map(try_toml_value_to_1darray::<f64>).map_or(pbc.clone(), |x| x.expect("invalid bc dim n"));
    assert_eq!(n.len(), gpe_count);
    let h = map.get("h").map(try_toml_value_to_1darray::<f64>).map_or(pbc.clone(), |x| x.expect("invalid bc dim h"));
    assert_eq!(h.len(), gpe_count);
    let r = map.get("r").map(try_toml_value_to_1darray::<f64>).map_or(pbc.clone(), |x| x.expect("invalid bc dim r"));
    assert_eq!(r.len(), gpe_count);
    let ca = map.get("ca").map(try_toml_value_to_1darray::<f64>).map_or(pbc.clone(), |x| x.expect("invalid bc dim ca"));
    assert_eq!(ca.len(), gpe_count);
    let s = map.get("s").map(try_toml_value_to_1darray::<f64>).map_or(pbc.clone(), |x| x.expect("invalid bc dim s"));
    assert_eq!(s.len(), gpe_count);

    let i_ext = map.get("i_ext").map_or(Array2::zeros((num_timesteps, stn_count)), |x| {
      let function = py_function_toml_string_to_py_object(x);
      vectorize_i_ext_py(&function, dt, total_t, stn_count)
    });

    let i_app = map.get("i_app").map_or(Array2::zeros((num_timesteps, stn_count)), |x| {
      let function = py_function_toml_string_to_py_object(x);
      vectorize_i_ext_py(&function, dt, total_t, stn_count)
    });

    let c_g_g = map
      .get("c_g_g")
      .map(try_toml_value_to_2darray::<f64>)
      .map_or(Array2::zeros((gpe_count, gpe_count)), |x| x.expect("invalid bc for c_g_s"));
    assert_eq!(c_g_g.shape(), &[gpe_count, gpe_count]);

    let c_s_g = map
      .get("c_s_g")
      .map(try_toml_value_to_2darray::<f64>)
      .map_or(Array2::zeros((gpe_count, gpe_count)), |x| x.expect("invalid bc for c_g_s"));
    assert_eq!(c_s_g.shape(), &[stn_count, gpe_count]);

    Self { count: gpe_count, v, n, h, r, ca, s, c_g_g, c_s_g, i_ext, i_app }
  }
}

#[derive(Clone)]
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
  pub fn with_bcs(mut self, bc: GPePopulationBoundryConditions) -> Self {
    self.v.row_mut(0).assign(&bc.v);
    self.n.row_mut(0).assign(&bc.n);
    self.h.row_mut(0).assign(&bc.h);
    self.r.row_mut(0).assign(&bc.r);
    self.ca.row_mut(0).assign(&bc.ca);
    self.s.row_mut(0).assign(&bc.s);
    self.c_g_g.assign(&bc.c_g_g);
    self.c_s_g.assign(&bc.c_s_g);
    self.i_ext.assign(&bc.i_ext);
    self.i_app.assign(&bc.i_app);
    self
  }

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

  pub fn new(num_timesteps: usize, gpe_count: usize, stn_count: usize) -> Self {
    GPePopulation {
      v: Array2::zeros((num_timesteps, gpe_count)),
      n: Array2::zeros((num_timesteps, gpe_count)),
      h: Array2::zeros((num_timesteps, gpe_count)),
      r: Array2::zeros((num_timesteps, gpe_count)),
      ca: Array2::zeros((num_timesteps, gpe_count)),
      s: Array2::zeros((num_timesteps, gpe_count)),
      i_l: Array2::zeros((num_timesteps, gpe_count)),
      i_k: Array2::zeros((num_timesteps, gpe_count)),
      i_na: Array2::zeros((num_timesteps, gpe_count)),
      i_t: Array2::zeros((num_timesteps, gpe_count)),
      i_ca: Array2::zeros((num_timesteps, gpe_count)),
      i_ahp: Array2::zeros((num_timesteps, gpe_count)),
      i_s_g: Array2::zeros((num_timesteps, gpe_count)),
      i_g_g: Array2::zeros((num_timesteps, gpe_count)),
      i_ext: Array2::zeros((num_timesteps, gpe_count)),
      i_app: Array2::zeros((num_timesteps, gpe_count)),
      c_s_g: Array2::zeros((stn_count, gpe_count)),
      c_g_g: Array2::zeros((gpe_count, gpe_count)),
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
