use std::fmt::Debug;
use std::ops::{Add, Mul};

use log::debug;
use ndarray::{s, Array1, Array2, ArrayView1, Ix2, OwnedRepr, ViewRepr};
use ndarray::{ArrayBase, Ix1};
use struct_field_names_as_array::FieldNamesAsSlice;

use crate::parameters::GPeParameters;
use crate::stn::{tau_x, x_oo};
use crate::types::*;
use crate::util::*;

#[derive(Default)]
pub struct GPe;

impl Neuron for GPe {
  const TYPE: &'static str = "GPe";
}

pub type GPeConfig = NeuronConfig<GPe, GPeParameters, GPePopulationBoundryConditions>;

#[derive(FieldNamesAsSlice, Debug, Default)]
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
  pub w_g_g: Array2<f64>,
  pub c_g_g: Array2<f64>,
  pub w_s_g: Array2<f64>,
  pub c_s_g: Array2<f64>,
  pub w_str: Array2<f64>,
  pub c_str: Array2<f64>,
}

impl Build<GPe, Boundary> for GPePopulationBoundryConditions {
  const PYTHON_CALLABLE_FIELD_NAMES: &[&'static str] = &["i_ext", "i_app"];
}

pub type BuilderGPeBoundary = Builder<GPe, Boundary, GPePopulationBoundryConditions>;

impl BuilderGPeBoundary {
  pub fn finish(
    self,
    gpe_count: usize,
    stn_count: usize,
    str_count: usize,
    dt: f64,
    total_t: f64,
    edge_resolution: u8,
  ) -> GPePopulationBoundryConditions {
    let pbc = Array1::zeros(gpe_count);

    let v =
      self.map.get("v").map(try_toml_value_to_1darray::<f64>).map_or(pbc.clone(), |x| x.expect("invalid bc dim v"));
    assert_eq!(v.len(), gpe_count);
    let n =
      self.map.get("n").map(try_toml_value_to_1darray::<f64>).map_or(pbc.clone(), |x| x.expect("invalid bc dim n"));
    assert_eq!(n.len(), gpe_count);
    let h =
      self.map.get("h").map(try_toml_value_to_1darray::<f64>).map_or(pbc.clone(), |x| x.expect("invalid bc dim h"));
    assert_eq!(h.len(), gpe_count);
    let r =
      self.map.get("r").map(try_toml_value_to_1darray::<f64>).map_or(pbc.clone(), |x| x.expect("invalid bc dim r"));
    assert_eq!(r.len(), gpe_count);
    let ca =
      self.map.get("ca").map(try_toml_value_to_1darray::<f64>).map_or(pbc.clone(), |x| x.expect("invalid bc dim ca"));
    assert_eq!(ca.len(), gpe_count);
    let s =
      self.map.get("s").map(try_toml_value_to_1darray::<f64>).map_or(pbc.clone(), |x| x.expect("invalid bc dim s"));
    assert_eq!(s.len(), gpe_count);

    let i_ext_f =
      toml_py_function_qualname_to_py_object(self.map.get("i_ext").expect("default should be set by caller"));
    let i_ext = vectorize_i_ext_py(&i_ext_f, dt / (edge_resolution as f64), total_t, gpe_count);

    let i_app_f =
      toml_py_function_qualname_to_py_object(self.map.get("i_app").expect("default should be set by caller"));
    let i_app = vectorize_i_ext_py(&i_app_f, dt / (edge_resolution as f64), total_t, gpe_count);

    debug!("GPe I_ext vectorized to\n{i_ext}");
    debug!("GPe I_app vectorized to\n{i_app}");

    let w_g_g = self
      .map
      .get("w_g_g")
      .map(try_toml_value_to_2darray::<f64>)
      .map_or(Array2::zeros((gpe_count, gpe_count)), |x| x.expect("invalid bc for w_g_g"));
    let c_g_g = self
      .map
      .get("c_g_g")
      .map(try_toml_value_to_2darray::<f64>)
      .map_or(w_g_g.mapv(|x| (x != 0.) as u8 as f64), |x| x.expect("invalid bc for c_g_g"))
      .mapv(|x| (x != 0.) as u8 as f64);
    assert_eq!(c_g_g.shape(), &[gpe_count, gpe_count]);

    let w_s_g = self
      .map
      .get("w_s_g")
      .map(try_toml_value_to_2darray::<f64>)
      .map_or(Array2::zeros((stn_count, gpe_count)), |x| x.expect("invalid bc for w_s_g"));
    let c_s_g = self
      .map
      .get("c_s_g")
      .map(try_toml_value_to_2darray::<f64>)
      .map_or(w_s_g.mapv(|x| (x != 0.) as u8 as f64), |x| x.expect("invalid bc for c_s_g"))
      .mapv(|x| (x != 0.) as u8 as f64);
    assert_eq!(c_s_g.shape(), &[stn_count, gpe_count]);

    let w_str = self
      .map
      .get("w_str")
      .map(try_toml_value_to_2darray::<f64>)
      .map_or(Array2::zeros((str_count, gpe_count)), |x| x.expect("invalid bc for w_g_g"));
    let c_str = self
      .map
      .get("c_str")
      .map(try_toml_value_to_2darray::<f64>)
      .map_or(w_str.mapv(|x| (x != 0.) as u8 as f64), |x| x.expect("invalid bc for c_g_g"))
      .mapv(|x| (x != 0.) as u8 as f64);
    assert_eq!(c_str.shape(), &[gpe_count, gpe_count]);

    GPePopulationBoundryConditions {
      count: gpe_count,
      v,
      n,
      h,
      r,
      ca,
      s,
      c_g_g,
      w_g_g,
      c_s_g,
      w_s_g,
      w_str,
      c_str,
      i_ext,
      i_app,
    }
  }
}

impl GPePopulationBoundryConditions {
  pub fn to_toml(&self, i_ext_py_qualified_name: &str, i_app_py_qualified_name: &str) -> toml::Value {
    let mut table = toml::value::Table::new();
    table.insert("count".to_owned(), (self.count as i64).into());
    table.insert("v".to_owned(), self.v.to_vec().into());
    table.insert("n".to_owned(), self.n.to_vec().into());
    table.insert("h".to_owned(), self.h.to_vec().into());
    table.insert("r".to_owned(), self.r.to_vec().into());
    table.insert("ca".to_owned(), self.ca.to_vec().into());
    table.insert("s".to_owned(), self.s.to_vec().into());
    table.insert("w_g_g".to_owned(), self.w_g_g.rows().into_iter().map(|x| x.to_vec()).collect::<Vec<_>>().into());
    table.insert("w_s_g".to_owned(), self.w_s_g.rows().into_iter().map(|x| x.to_vec()).collect::<Vec<_>>().into());
    table.insert("i_ext".to_owned(), toml::Value::String(i_ext_py_qualified_name.to_owned()));
    table.insert("i_app".to_owned(), toml::Value::String(i_app_py_qualified_name.to_owned()));

    toml::Value::Table(table)
  }
}

pub struct GPeState<T>
where
  T: ndarray::Data<Elem = f64>,
{
  pub v: ArrayBase<T, Ix1>,
  pub n: ArrayBase<T, Ix1>,
  pub h: ArrayBase<T, Ix1>,
  pub r: ArrayBase<T, Ix1>,
  pub ca: ArrayBase<T, Ix1>,
  pub s: ArrayBase<T, Ix1>,
  pub w_g_g: ArrayBase<T, Ix2>,
  pub w_s_g: ArrayBase<T, Ix2>,
  pub w_str: ArrayBase<T, Ix2>,
  pub ca_g_g: ArrayBase<T, Ix2>,
  pub ca_s_g: ArrayBase<T, Ix2>,
  pub ca_str: ArrayBase<T, Ix2>,
}

impl<T> Debug for GPeState<T>
where
  T: ndarray::Data<Elem = f64> + Debug,
{
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_struct("GPeState")
      .field("v", &self.v)
      .field("n", &self.n)
      .field("r", &self.r)
      .field("ca", &self.ca)
      .field("s", &self.s)
      .field("w_g_g", &self.w_g_g)
      .field("w_s_g", &self.w_s_g)
      .field("ca_g_g", &self.ca_g_g)
      .field("ca_s_g", &self.ca_s_g)
      .finish()
  }
}

impl<T> GPeState<T>
where
  T: ndarray::Data<Elem = f64>,
{
  pub fn dydt(
    &self,
    p: &GPeParameters,
    d_gpe: &ArrayView1<f64>,
    d_stn: &ArrayView1<f64>,
    d_str: &ArrayView1<f64>,
    s_stn: &ArrayView1<f64>,
    s_str: &ArrayView1<f64>,
    i_ext: &ArrayView1<f64>,
    i_app: &ArrayView1<f64>,
  ) -> GPeState<OwnedRepr<f64>> {
    let Self { v, n, h, r, ca, s, w_g_g, w_s_g, w_str, ca_g_g, ca_s_g, ca_str } = self;

    let n_oo = x_oo(v, p.tht_n, p.sig_n);
    let m_oo = x_oo(v, p.tht_m, p.sig_m);
    let h_oo = x_oo(v, p.tht_h, p.sig_h);
    let a_oo = x_oo(v, p.tht_a, p.sig_a);
    let r_oo = x_oo(v, p.tht_r, p.sig_r);
    let s_oo = x_oo(v, p.tht_s, p.sig_s);
    let h_syn_oo = x_oo(&(v - p.tht_g), p.tht_g_h, p.sig_g_h);

    let tau_n = tau_x(v, p.tau_n_0, p.tau_n_1, p.tht_n_t, p.sig_n_t);
    let tau_h = tau_x(v, p.tau_h_0, p.tau_h_1, p.tht_h_t, p.sig_h_t);

    let i_l = p.g_l * (v - p.v_l);
    let i_k = p.g_k * n.powi(4) * (v - p.v_k);
    let i_na = p.g_na * m_oo.powi(3) * h * (v - p.v_na);
    let i_t = p.g_t * a_oo.powi(3) * r * (v - p.v_ca);
    let i_ca = p.g_ca * s_oo.powi(2) * (v - p.v_ca);
    let i_ahp = p.g_ahp * (v - p.v_k) * ca / (ca + p.k_1);
    let i_s_g = p.g_s_g * (v - p.v_s_g) * (self.w_s_g.t().dot(s_stn));
    let i_g_g = p.g_g_g * (v - p.v_g_g) * (self.w_g_g.t().dot(s));
    let i_str = p.g_str * (v - p.v_str) * (self.w_str.t().dot(s_str));

    // Update state
    let dy = GPeState {
      v: -i_l - i_k - i_na - &i_t - &i_ca - i_ahp - i_s_g - i_g_g - i_str - i_ext + i_app,
      n: p.phi_n * (n_oo - n) / tau_n,
      h: p.phi_h * (h_oo - h) / tau_h,
      r: p.phi_r * (r_oo - r) / p.tau_r,
      ca: p.eps * ((-i_ca - i_t) - p.k_ca * ca),
      s: p.alpha * h_syn_oo * (1. - s) - p.beta * s,
      w_g_g: ndarray::Array::zeros(w_g_g.raw_dim()), // TODO - no plasticity
      w_s_g: ndarray::Array::zeros(w_s_g.raw_dim()), // TODO - no plasticity
      w_str: ndarray::Array::zeros(w_str.raw_dim()), // TODO - no plasticity
      ca_g_g: -ca_g_g / p.tau_ca + p.ca_pre * &d_gpe.iax(1) + p.ca_post * &d_gpe.iax(0),
      ca_s_g: -ca_s_g / p.tau_ca + p.ca_pre * &d_stn.iax(1) + p.ca_post * &d_gpe.iax(0),
      ca_str: -ca_str / p.tau_ca + p.ca_pre * &d_str.iax(1) + p.ca_post * &d_gpe.iax(0),
    };

    dy
  }
}

impl<'l, 'r, R, L> Add<&'r GPeState<R>> for &'l GPeState<L>
where
  R: ndarray::Data<Elem = f64>,
  L: ndarray::Data<Elem = f64>,
{
  type Output = GPeState<OwnedRepr<f64>>;
  fn add(self, rhs: &'r GPeState<R>) -> Self::Output {
    GPeState {
      v: &self.v + &rhs.v,
      n: &self.n + &rhs.n,
      h: &self.h + &rhs.h,
      r: &self.r + &rhs.r,
      ca: &self.ca + &rhs.ca,
      s: &self.s + &rhs.s,
      w_g_g: &self.w_g_g + &rhs.w_g_g,
      w_s_g: &self.w_s_g + &rhs.w_s_g,
      w_str: &self.w_str + &rhs.w_str,
      ca_g_g: &self.ca_g_g + &rhs.ca_g_g,
      ca_s_g: &self.ca_s_g + &rhs.ca_s_g,
      ca_str: &self.ca_str + &rhs.ca_str,
    }
  }
}

impl<'r, R, L> Add<&'r GPeState<R>> for GPeState<L>
where
  R: ndarray::Data<Elem = f64>,
  L: ndarray::Data<Elem = f64>,
{
  type Output = GPeState<OwnedRepr<f64>>;
  fn add(self, rhs: &'r GPeState<R>) -> Self::Output {
    &self + rhs
  }
}

impl<'l, R, L> Add<GPeState<R>> for &'l GPeState<L>
where
  R: ndarray::Data<Elem = f64>,
  L: ndarray::Data<Elem = f64>,
{
  type Output = GPeState<OwnedRepr<f64>>;
  fn add(self, rhs: GPeState<R>) -> Self::Output {
    self + &rhs
  }
}

impl<R, L> Add<GPeState<R>> for GPeState<L>
where
  R: ndarray::Data<Elem = f64>,
  L: ndarray::Data<Elem = f64>,
{
  type Output = GPeState<OwnedRepr<f64>>;
  fn add(self, rhs: GPeState<R>) -> Self::Output {
    &self + &rhs
  }
}

impl<'a, T> Mul<&'a GPeState<T>> for f64
where
  T: ndarray::Data<Elem = f64>,
{
  type Output = GPeState<OwnedRepr<f64>>;
  fn mul(self, rhs: &'a GPeState<T>) -> Self::Output {
    //   let mut w_s_g = Array2::zeros(rhs.w_s_g.raw_dim());
    //   let mut ca_s_g = Array2::zeros(rhs.ca_s_g.raw_dim());
    //   rhs.a_s_g.iter().for_each(|&(i, j)| {
    //     w_s_g[[i, j]] = self * rhs.w_s_g[[i, j]];
    //     ca_s_g[[i, j]] = self * rhs.ca_s_g[[i, j]];
    //   });
    //   let mut w_g_g = Array2::zeros(rhs.w_g_g.raw_dim());
    //   let mut ca_g_g = Array2::zeros(rhs.ca_g_g.raw_dim());
    //   rhs.a_g_g.iter().for_each(|&(i, j)| {
    //     w_g_g[[i, j]] = self * rhs.w_g_g[[i, j]];
    //     ca_g_g[[i, j]] = self * rhs.ca_g_g[[i, j]];
    //   });

    GPeState {
      v: self * &rhs.v,
      n: self * &rhs.n,
      h: self * &rhs.h,
      r: self * &rhs.r,
      ca: self * &rhs.ca,
      s: self * &rhs.s,
      w_g_g: self * &rhs.w_g_g,
      w_s_g: self * &rhs.w_s_g,
      w_str: self * &rhs.w_str,
      ca_g_g: self * &rhs.ca_g_g,
      ca_s_g: self * &rhs.ca_s_g,
      ca_str: self * &rhs.ca_str,
    }
  }
}

impl<T> Mul<GPeState<T>> for f64
where
  T: ndarray::Data<Elem = f64>,
{
  type Output = GPeState<OwnedRepr<f64>>;
  fn mul(self, rhs: GPeState<T>) -> Self::Output {
    self * &rhs
  }
}

impl<'a, T> Mul<f64> for &'a GPeState<T>
where
  T: ndarray::Data<Elem = f64>,
{
  type Output = GPeState<OwnedRepr<f64>>;
  fn mul(self, rhs: f64) -> Self::Output {
    rhs * self
  }
}

impl<'a, T> Mul<f64> for GPeState<T>
where
  T: ndarray::Data<Elem = f64>,
{
  type Output = GPeState<OwnedRepr<f64>>;
  fn mul(self, rhs: f64) -> Self::Output {
    rhs * &self
  }
}

#[derive(Default, Clone)]
pub struct GPeHistory {
  pub v: Array2<f64>,
  pub n: Array2<f64>,
  pub h: Array2<f64>,
  pub r: Array2<f64>,
  pub ca: Array2<f64>,
  pub s: Array2<f64>,
  pub w_s_g: Array2<f64>,
  pub w_g_g: Array2<f64>,
  pub w_str: Array2<f64>,
  pub ca_s_g: Array2<f64>,
  pub ca_g_g: Array2<f64>,
  pub ca_str: Array2<f64>,
  pub i_ext: Array2<f64>,
  pub i_app: Array2<f64>,
}

impl GPeHistory {
  pub fn new(num_timesteps: usize, gpe_count: usize, stn_count: usize, edge_resolution: usize) -> Self {
    Self {
      v: Array2::zeros((num_timesteps, gpe_count)),
      n: Array2::zeros((num_timesteps, gpe_count)),
      h: Array2::zeros((num_timesteps, gpe_count)),
      r: Array2::zeros((num_timesteps, gpe_count)),
      ca: Array2::zeros((num_timesteps, gpe_count)),
      s: Array2::zeros((num_timesteps, gpe_count)),
      i_ext: Array2::zeros((num_timesteps * edge_resolution, gpe_count)),
      i_app: Array2::zeros((num_timesteps * edge_resolution, gpe_count)),
      w_s_g: Array2::zeros((stn_count, gpe_count)),
      w_g_g: Array2::zeros((gpe_count, gpe_count)),
      w_str: Array2::zeros((gpe_count, gpe_count)),
      ca_s_g: Array2::zeros((stn_count, gpe_count)),
      ca_g_g: Array2::zeros((gpe_count, gpe_count)),
      ca_str: Array2::zeros((gpe_count, gpe_count)),
    }
  }

  pub fn with_bcs(mut self, bc: GPePopulationBoundryConditions) -> Self {
    self.v.row_mut(0).assign(&bc.v);
    self.n.row_mut(0).assign(&bc.n);
    self.h.row_mut(0).assign(&bc.h);
    self.r.row_mut(0).assign(&bc.r);
    self.ca.row_mut(0).assign(&bc.ca);
    self.s.row_mut(0).assign(&bc.s);
    self.w_g_g.assign(&bc.w_g_g);
    self.w_s_g.assign(&bc.w_s_g);
    self.w_str.assign(&bc.w_str);
    self.i_ext.assign(&bc.i_ext);
    self.i_app.assign(&bc.i_app);
    self
  }

  pub fn insert(&mut self, it: usize, y: &GPeState<OwnedRepr<f64>>) {
    self.v.row_mut(it).assign(&y.v);
    self.n.row_mut(it).assign(&y.n);
    self.r.row_mut(it).assign(&y.r);
    self.h.row_mut(it).assign(&y.h);
    self.ca.row_mut(it).assign(&y.ca);
    self.s.row_mut(it).assign(&y.s);
    self.w_s_g = y.w_s_g.clone();
    self.w_g_g = y.w_g_g.clone();
    self.ca_s_g = y.ca_s_g.clone();
    self.ca_g_g = y.ca_g_g.clone();
  }

  pub fn row<'a>(&'a self, it: usize) -> GPeState<ViewRepr<&'a f64>> {
    GPeState {
      v: self.v.row(it),
      n: self.n.row(it),
      h: self.h.row(it),
      r: self.r.row(it),
      ca: self.ca.row(it),
      s: self.s.row(it),
      w_s_g: self.w_s_g.view(),
      w_g_g: self.w_g_g.view(),
      w_str: self.w_str.view(),
      ca_s_g: self.ca_s_g.view(),
      ca_g_g: self.ca_g_g.view(),
      ca_str: self.ca_str.view(),
    }
  }

  pub fn into_compressed_polars_df(
    &self,
    idt: f64,
    odt: Option<f64>,
    edge_resolution: usize,
  ) -> polars::prelude::DataFrame {
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
    let erange = s![0..num_timesteps * edge_resolution;step * edge_resolution, ..];

    let num_timesteps = self.v.slice(srange).nrows();

    let time = (ndarray::Array1::range(0., num_timesteps as f64, 1.) * output_dt)
      .to_shape((num_timesteps, 1))
      .unwrap()
      .to_owned();

    polars::prelude::DataFrame::new(vec![
      array2_to_polars_column("time", time.view()),
      array2_to_polars_column("v", self.v.slice(srange)),
      array2_to_polars_column("n", self.n.slice(srange)),
      array2_to_polars_column("h", self.h.slice(srange)),
      array2_to_polars_column("r", self.r.slice(srange)),
      array2_to_polars_column("ca", self.ca.slice(srange)),
      array2_to_polars_column("s", self.s.slice(srange)),
      array2_to_polars_column("i_ext", self.i_ext.slice(erange)),
      array2_to_polars_column("i_app", self.i_app.slice(erange)),
      unit_to_polars_column("w_g_g", self.w_g_g.view(), num_timesteps),
      unit_to_polars_column("w_s_g", self.w_s_g.view(), num_timesteps),
    ])
    .expect("This shouldn't happend if the struct is valid")
  }
}
