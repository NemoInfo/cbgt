use log::debug;
use ndarray::{s, Array1, Array2, ArrayView1};
use struct_field_names_as_array::FieldNamesAsArray;

use crate::parameters::STNParameters;
use crate::util::*;
use crate::ModelDescription;
use crate::EXPERIMENT_BC_FILE_NAME;

#[derive(FieldNamesAsArray, Debug)]
pub struct STNPopulationBoundryConditions {
  pub count: usize,
  // State
  pub v: Array1<f64>,
  pub n: Array1<f64>,
  pub h: Array1<f64>,
  pub r: Array1<f64>,
  pub ca: Array1<f64>,
  pub s: Array1<f64>,
  pub i_ext: Array2<f64>,

  // Connection Matrice
  pub c_g_s: Array2<f64>,
}

impl ModelDescription for STNPopulationBoundryConditions {
  const TYPE: &'static str = "STN";
  const EXPERIMENT_FILE_NAME: &'static str = EXPERIMENT_BC_FILE_NAME;
  const DEFAULT_PATH: Option<&'static str> = None;
}

impl STNPopulationBoundryConditions {
  pub fn to_toml(&self, i_ext_py_qualified_name: &str) -> toml::Value {
    let mut table = toml::value::Table::new();
    table.insert("count".to_owned(), (self.count as i64).into());
    table.insert("v".to_owned(), self.v.to_vec().into());
    table.insert("n".to_owned(), self.n.to_vec().into());
    table.insert("h".to_owned(), self.h.to_vec().into());
    table.insert("r".to_owned(), self.r.to_vec().into());
    table.insert("ca".to_owned(), self.ca.to_vec().into());
    table.insert("s".to_owned(), self.s.to_vec().into());
    table.insert("c_g_s".to_owned(), self.c_g_s.rows().into_iter().map(|x| x.to_vec()).collect::<Vec<_>>().into());
    table.insert("i_ext".to_owned(), toml::Value::String(i_ext_py_qualified_name.to_owned()));

    toml::Value::Table(table)
  }

  pub fn from(
    map: toml::map::Map<String, toml::Value>,
    stn_count: usize,
    gpe_count: usize,
    dt: f64,
    total_t: f64,
  ) -> Self {
    let pbc = Array1::zeros(stn_count); // TODO wathafuq

    let v = map.get("v").map(try_toml_value_to_1darray::<f64>).map_or(pbc.clone(), |x| x.expect("invalid bc dim v"));
    assert_eq!(v.len(), stn_count);
    let n = map.get("n").map(try_toml_value_to_1darray::<f64>).map_or(pbc.clone(), |x| x.expect("invalid bc dim n"));
    assert_eq!(n.len(), stn_count);
    let h = map.get("h").map(try_toml_value_to_1darray::<f64>).map_or(pbc.clone(), |x| x.expect("invalid bc dim h"));
    assert_eq!(h.len(), stn_count);
    let r = map.get("r").map(try_toml_value_to_1darray::<f64>).map_or(pbc.clone(), |x| x.expect("invalid bc dim r"));
    assert_eq!(r.len(), stn_count);
    let ca = map.get("ca").map(try_toml_value_to_1darray::<f64>).map_or(pbc.clone(), |x| x.expect("invalid bc dim ca"));
    assert_eq!(ca.len(), stn_count);
    let s = map.get("s").map(try_toml_value_to_1darray::<f64>).map_or(pbc.clone(), |x| x.expect("invalid bc dim s"));
    assert_eq!(s.len(), stn_count);

    let i_ext_f = toml_py_function_qualname_to_py_object(map.get("i_ext").expect("default should be set by caller"));
    let i_ext = vectorize_i_ext_py(&i_ext_f, dt, total_t, stn_count);

    debug!("STN I_ext vectorized to\n{i_ext}");

    let c_g_s = map
      .get("c_g_s")
      .map(try_toml_value_to_2darray::<f64>)
      .map_or(Array2::zeros((gpe_count, stn_count)), |x| x.expect("invalid bc for c_g_s"));
    assert_eq!(c_g_s.shape(), &[gpe_count, stn_count]);
    assert_eq!(i_ext.shape()[1], stn_count);

    Self { count: stn_count, v, n, h, r, ca, s, c_g_s, i_ext }
  }
}

#[derive(Clone)]
pub struct STNPopulation {
  // TODO add count here
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

  // Connection Matrice
  pub c_g_s: Array2<f64>,

  // Connection Currents
  pub i_g_s: Array2<f64>,
}

impl STNPopulation {
  pub fn with_bcs(mut self, bc: STNPopulationBoundryConditions) -> Self {
    self.v.row_mut(0).assign(&bc.v);
    self.n.row_mut(0).assign(&bc.n);
    self.h.row_mut(0).assign(&bc.h);
    self.r.row_mut(0).assign(&bc.r);
    self.ca.row_mut(0).assign(&bc.ca);
    self.s.row_mut(0).assign(&bc.s);
    self.c_g_s.assign(&bc.c_g_s);
    self.i_ext.assign(&bc.i_ext);
    self
  }

  pub fn new(num_timesteps: usize, stn_count: usize, gpe_count: usize) -> Self {
    Self {
      v: Array2::zeros((num_timesteps, stn_count)),
      n: Array2::zeros((num_timesteps, stn_count)),
      h: Array2::zeros((num_timesteps, stn_count)),
      r: Array2::zeros((num_timesteps, stn_count)),
      ca: Array2::zeros((num_timesteps, stn_count)),
      s: Array2::zeros((num_timesteps, stn_count)),
      i_l: Array2::zeros((num_timesteps, stn_count)),
      i_k: Array2::zeros((num_timesteps, stn_count)),
      i_na: Array2::zeros((num_timesteps, stn_count)),
      i_t: Array2::zeros((num_timesteps, stn_count)),
      i_ca: Array2::zeros((num_timesteps, stn_count)),
      i_ahp: Array2::zeros((num_timesteps, stn_count)),
      i_g_s: Array2::zeros((num_timesteps, stn_count)),
      i_ext: Array2::zeros((num_timesteps, stn_count)),
      c_g_s: Array2::zeros((gpe_count, stn_count)),
    }
  }

  pub fn euler_step(&mut self, it: usize, dt: f64, stn: &STNParameters, s_gpe: &ArrayView1<f64>) {
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
    let mut i_g_s = self.i_g_s.slice_mut(t);

    i_l.assign(&(stn.g_l * (v - stn.v_l)));
    i_k.assign(&(stn.g_k * n.powi(4) * (v - stn.v_k)));
    i_na.assign(&(stn.g_na * m_inf.powi(3) * h * (v - stn.v_na)));
    i_t.assign(&(stn.g_t * a_inf.powi(3) * b_inf.pow2() * (v - stn.v_ca)));
    i_ca.assign(&(stn.g_ca * s_inf.powi(2) * (v - stn.v_ca)));
    i_ahp.assign(&(stn.g_ahp * (v - stn.v_k) * ca / (ca + stn.k_1)));
    i_g_s.assign(&(stn.g_g_s * (v - stn.v_g_s) * (self.c_g_s.t().dot(s_gpe))));

    // Update state
    v1.assign(&(v + dt * (-&i_l - &i_k - &i_na - &i_t - &i_ca - &i_ahp - &i_g_s - &self.i_ext.row(it))));
    n1.assign(&(n + dt * stn.phi_n * (n_inf - n) / tau_n));
    h1.assign(&(h + dt * stn.phi_h * (h_inf - h) / tau_h));
    r1.assign(&(r + dt * stn.phi_r * (r_inf - r) / tau_r));
    ca1.assign(&(ca + dt * stn.eps * ((-&i_ca - &i_t) - stn.k_ca * ca)));

    // Update synapses
    let h_syn_inf = x_inf(&(v - stn.tht_g).view(), stn.tht_g_h, stn.sig_g_h);
    s1.assign(&(s + dt * (stn.alpha * h_syn_inf * (1. - s) - stn.beta * s)));
  }

  pub fn into_compressed_polars_df(&self, idt: f64, odt: Option<f64>) -> polars::prelude::DataFrame {
    let num_timesteps = self.v.nrows();
    let odt = odt.unwrap_or(1.); // ms
    let step = odt / idt;
    if step != step.trunc() {
      log::warn!(
        "output_dt / simulation_dt = {step} is not integer. With a step of {} => output_dt = {}",
        step.trunc(),
        step.trunc() * idt
      );
    }
    let step = step.trunc() as usize;
    let output_dt = step as f64 * idt;
    let srange = s![0..num_timesteps;step, ..];

    let num_timesteps = self.v.slice(srange).nrows();

    let time = (ndarray::Array1::range(0., num_timesteps as f64, 1.) * output_dt)
      .to_shape((num_timesteps, 1))
      .unwrap()
      .to_owned();

    polars::prelude::DataFrame::new(vec![
      array2_to_polars_column("time", time.view()),
      array2_to_polars_column("v", self.v.slice(srange)),
      array2_to_polars_column("i_l", self.i_l.slice(srange)),
      array2_to_polars_column("i_k", self.i_k.slice(srange)),
      array2_to_polars_column("i_na", self.i_na.slice(srange)),
      array2_to_polars_column("i_t", self.i_t.slice(srange)),
      array2_to_polars_column("i_ca", self.i_ca.slice(srange)),
      array2_to_polars_column("i_ahp", self.i_ahp.slice(srange)),
      array2_to_polars_column("i_g_s", self.i_g_s.slice(srange)),
      array2_to_polars_column("i_ext", self.i_ext.slice(srange)),
      array2_to_polars_column("s", self.s.slice(srange)),
    ])
    .expect("This shouldn't happend if the struct is valid")
  }
}
