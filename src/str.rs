use std::fmt::Debug;
use std::ops::Add;
use std::ops::Mul;

use log::debug;
use ndarray::{s, Array1, Array2, ArrayBase, ArrayView1, Ix1, Ix2, OwnedRepr, ViewRepr};
use struct_field_names_as_array::FieldNamesAsSlice;

use crate::parameters::STRParameters;
use crate::types::*;
use crate::util::*;

#[derive(Default, Debug)]
pub struct STR;

impl Neuron for STR {
  const TYPE: &'static str = "STR";
}

pub type STRConfig = NeuronConfig<STR, STRParameters, STRPopulationBoundryConditions>;

#[derive(FieldNamesAsSlice, Debug, Default)]
pub struct STRPopulationBoundryConditions {
  pub count: usize,
  // State
  pub v: Array1<f64>,
  pub n: Array1<f64>,
  pub h: Array1<f64>,
  pub r: Array1<f64>,
  pub s: Array1<f64>,
  pub w_str: Array2<f64>,
  pub w_ctx: Array2<f64>,
  pub i_ext: Array2<f64>,
}

impl Build<STR, Boundary> for STRPopulationBoundryConditions {
  const PYTHON_CALLABLE_FIELD_NAMES: &[&'static str] = &["i_ext"];
}

pub type BuilderSTRBoundary = Builder<STR, Boundary, STRPopulationBoundryConditions>;

impl Builder<STR, Boundary, STRPopulationBoundryConditions> {
  pub fn finish(
    self,
    str_count: usize,
    ctx_count: usize,
    dt: f64,
    total_t: f64,
    edge_resolution: u8,
  ) -> STRPopulationBoundryConditions {
    // TODO: Decide what should happen with the defaults?
    // Instead of zero it should panic if top level use_default is set to false
    let zero = Array1::zeros(str_count);

    let v =
      self.map.get("v").map(try_toml_value_to_1darray::<f64>).map_or(zero.clone(), |x| x.expect("invalid bc dim v"));
    assert_eq!(v.len(), str_count);
    let n =
      self.map.get("n").map(try_toml_value_to_1darray::<f64>).map_or(zero.clone(), |x| x.expect("invalid bc dim n"));
    assert_eq!(n.len(), str_count);
    let h =
      self.map.get("h").map(try_toml_value_to_1darray::<f64>).map_or(zero.clone(), |x| x.expect("invalid bc dim h"));
    assert_eq!(h.len(), str_count);
    let r =
      self.map.get("r").map(try_toml_value_to_1darray::<f64>).map_or(zero.clone(), |x| x.expect("invalid bc dim r"));
    assert_eq!(r.len(), str_count);
    let ca =
      self.map.get("ca").map(try_toml_value_to_1darray::<f64>).map_or(zero.clone(), |x| x.expect("invalid bc dim ca"));
    assert_eq!(ca.len(), str_count);
    let s =
      self.map.get("s").map(try_toml_value_to_1darray::<f64>).map_or(zero.clone(), |x| x.expect("invalid bc dim s"));
    assert_eq!(s.len(), str_count);

    let i_ext_f =
      toml_py_function_qualname_to_py_object(self.map.get("i_ext").expect("default should be set by caller"));
    let i_ext = vectorize_i_ext_py(&i_ext_f, dt / (edge_resolution as f64), total_t, str_count);
    assert_eq!(i_ext.shape()[1], str_count);
    debug!("STR I_ext vectorized to\n{i_ext}");

    let w_str = self
      .map
      .get("w_str")
      .map(try_toml_value_to_2darray::<f64>)
      .map_or(Array2::zeros((str_count, str_count)), |x| x.expect("invalid bc for w_str_str"));
    let c_str = self
      .map
      .get("c_str")
      .map(try_toml_value_to_2darray::<f64>)
      .map_or(w_str.mapv(|x| (x != 0.) as u8 as f64), |x| x.expect("invalid bc for c_str_str"))
      .mapv(|x| (x != 0.) as u8 as f64);
    assert_eq!(c_str.shape(), &[str_count, str_count]);

    let w_ctx = self
      .map
      .get("w_ctx")
      .map(try_toml_value_to_2darray::<f64>)
      .map_or(Array2::zeros((ctx_count, str_count)), |x| x.expect("invalid bc for w_ctx_str"));
    let c_ctx = self
      .map
      .get("c_ctx")
      .map(try_toml_value_to_2darray::<f64>)
      .map_or(w_ctx.mapv(|x| (x != 0.) as u8 as f64), |x| x.expect("invalid bc for c_ctx_str"))
      .mapv(|x| (x != 0.) as u8 as f64);
    assert_eq!(c_ctx.shape(), &[ctx_count, str_count]);

    STRPopulationBoundryConditions { count: str_count, v, n, h, r, s, w_str, w_ctx, i_ext }
  }
}

impl STRPopulationBoundryConditions {
  pub fn to_toml(&self, i_ext_py_qualified_name: &str) -> toml::Value {
    let mut table = toml::value::Table::new();
    table.insert("count".to_owned(), (self.count as i64).into());
    table.insert("v".to_owned(), self.v.to_vec().into());
    table.insert("n".to_owned(), self.n.to_vec().into());
    table.insert("h".to_owned(), self.h.to_vec().into());
    table.insert("r".to_owned(), self.r.to_vec().into());
    table.insert("s".to_owned(), self.s.to_vec().into());
    table.insert("i_ext".to_owned(), toml::Value::String(i_ext_py_qualified_name.to_owned()));

    toml::Value::Table(table)
  }
}

#[derive(Default, Clone)]
pub struct STRHistory {
  pub v: Array2<f64>,
  pub n: Array2<f64>,
  pub h: Array2<f64>,
  pub r: Array2<f64>,
  pub s: Array2<f64>,
  pub i_ext: Array2<f64>,
  pub w_str: Array2<f64>,
  pub ca_syn_str: Array2<f64>,
  pub w_ctx: Array2<f64>,
}

impl STRHistory {
  pub fn new(num_timesteps: usize, str_count: usize, ctx_count: usize, edge_resolution: usize) -> Self {
    Self {
      v: Array2::zeros((num_timesteps, str_count)),
      n: Array2::zeros((num_timesteps, str_count)),
      h: Array2::zeros((num_timesteps, str_count)),
      r: Array2::zeros((num_timesteps, str_count)),
      s: Array2::zeros((num_timesteps, str_count)),
      w_str: Array2::zeros((str_count, str_count)),
      ca_syn_str: Array2::zeros((str_count, str_count)),
      i_ext: Array2::zeros((num_timesteps * edge_resolution, str_count)),
      w_ctx: Array2::zeros((ctx_count, ctx_count)),
    }
  }

  pub fn with_bcs(mut self, bc: STRPopulationBoundryConditions) -> Self {
    self.v.row_mut(0).assign(&bc.v);
    self.n.row_mut(0).assign(&bc.n);
    self.h.row_mut(0).assign(&bc.h);
    self.r.row_mut(0).assign(&bc.r);
    self.s.row_mut(0).assign(&bc.s);
    self.w_str.assign(&bc.w_str);
    self.w_ctx.assign(&bc.w_ctx);
    self.i_ext.assign(&bc.i_ext);
    self
  }

  pub fn insert(&mut self, it: usize, y: &STRState<OwnedRepr<f64>>) {
    self.v.row_mut(it).assign(&y.v);
    self.n.row_mut(it).assign(&y.n);
    self.r.row_mut(it).assign(&y.r);
    self.h.row_mut(it).assign(&y.h);
    self.s.row_mut(it).assign(&y.s);
  }

  pub fn row<'a>(&'a self, it: usize) -> STRState<ViewRepr<&'a f64>> {
    STRState {
      v: self.v.row(it),
      n: self.n.row(it),
      h: self.h.row(it),
      r: self.r.row(it),
      s: self.s.row(it),
      w_str: self.w_str.view(),
      ca_syn_str: self.ca_syn_str.view(),
      w_ctx: self.w_ctx.view(),
    }
  }
}

impl<'a, T> Mul<&'a STRState<T>> for f64
where
  T: ndarray::Data<Elem = f64>,
{
  type Output = STRState<OwnedRepr<f64>>;
  fn mul(self, rhs: &'a STRState<T>) -> Self::Output {
    STRState {
      v: self * &rhs.v,
      n: self * &rhs.n,
      h: self * &rhs.h,
      r: self * &rhs.r,
      s: self * &rhs.s,
      w_str: self * &rhs.w_str,
      ca_syn_str: self * &rhs.ca_syn_str,
      w_ctx: self * &rhs.w_ctx,
    }
  }
}

impl<T> Mul<STRState<T>> for f64
where
  T: ndarray::Data<Elem = f64>,
{
  type Output = STRState<OwnedRepr<f64>>;
  fn mul(self, rhs: STRState<T>) -> Self::Output {
    self * &rhs
  }
}

impl<'a, T> Mul<f64> for &'a STRState<T>
where
  T: ndarray::Data<Elem = f64>,
{
  type Output = STRState<OwnedRepr<f64>>;
  fn mul(self, rhs: f64) -> Self::Output {
    rhs * self
  }
}

impl<'a, T> Mul<f64> for STRState<T>
where
  T: ndarray::Data<Elem = f64>,
{
  type Output = STRState<OwnedRepr<f64>>;
  fn mul(self, rhs: f64) -> Self::Output {
    rhs * &self
  }
}

pub struct STRState<T>
where
  T: ndarray::RawData,
{
  pub v: ArrayBase<T, Ix1>,
  pub n: ArrayBase<T, Ix1>,
  pub h: ArrayBase<T, Ix1>,
  pub r: ArrayBase<T, Ix1>,
  pub s: ArrayBase<T, Ix1>,
  pub w_str: ArrayBase<T, Ix2>,
  pub ca_syn_str: ArrayBase<T, Ix2>,
  pub w_ctx: ArrayBase<T, Ix2>,
}

impl<T> Debug for STRState<T>
where
  T: ndarray::Data<Elem = f64>,
{
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_struct("STRState").field("v", &self.v).field("n", &self.n).field("r", &self.r).field("s", &self.s).finish()
  }
}

impl<T> STRState<T>
where
  T: ndarray::Data<Elem = f64>,
{
  pub fn dydt(
    &self,
    p: &STRParameters,
    s_ctx: &ArrayView1<f64>,
    i_ext: &ArrayView1<f64>,
  ) -> STRState<OwnedRepr<T::Elem>> {
    let Self { v, n, h, r, s, w_str, ca_syn_str, w_ctx } = self;

    let n_oo = x_oo(v, p.tht_n, p.sig_n);
    let m_oo = x_oo(v, p.tht_m, p.sig_m);
    let h_oo = x_oo(v, p.tht_h, p.sig_h);
    let r_oo = x_oo(v, p.tht_r, p.sig_r);
    let h_syn_oo = x_oo(&(v - p.tht_g), p.tht_g_h, p.sig_g_h);

    let tau_n = tau_x(v, p.tau_n_0, p.tau_n_1, p.tht_n_t, p.sig_n_t);
    let tau_h = tau_x(v, p.tau_h_0, p.tau_h_1, p.tht_h_t, p.sig_h_t);
    let tau_r = tau_x(v, p.tau_r_0, p.tau_r_1, p.tht_r_t, p.sig_r_t);

    let i_l = p.g_l * (v - p.v_l);
    let i_k = p.g_k * n.powi(4) * (v - p.v_k);
    let i_na = p.g_na * m_oo.powi(3) * h * (v - p.v_na);
    let i_str = p.g_str * (v - p.v_str) * (w_str.t().dot(s));
    let i_ctx = p.g_ctx * (v - p.v_ctx) * (w_ctx.t().dot(s_ctx));

    // TODO maybe add plasticity here

    STRState {
      v: -i_l - i_k - i_na - i_str - i_ctx - i_ext,
      n: p.phi_n * (n_oo - n) / tau_n,
      h: p.phi_h * (h_oo - h) / tau_h,
      r: p.phi_r * (r_oo - r) / tau_r,
      s: p.alpha * h_syn_oo * (1. - s) - p.beta * s,
      w_str: Array2::<f64>::zeros(w_str.raw_dim()),
      ca_syn_str: Array2::<f64>::zeros(ca_syn_str.raw_dim()),
      w_ctx: Array2::<f64>::zeros(w_ctx.raw_dim()),
    }
  }
}

impl<'l, 'r, R, L> Add<&'r STRState<R>> for &'l STRState<L>
where
  R: ndarray::Data<Elem = f64>,
  L: ndarray::Data<Elem = f64>,
{
  type Output = STRState<OwnedRepr<f64>>;
  fn add(self, rhs: &'r STRState<R>) -> Self::Output {
    STRState {
      v: &self.v + &rhs.v,
      n: &self.n + &rhs.n,
      h: &self.h + &rhs.h,
      r: &self.r + &rhs.r,
      s: &self.s + &rhs.s,
      w_str: &self.w_str + &rhs.w_str,
      ca_syn_str: &self.ca_syn_str + &rhs.ca_syn_str,
      w_ctx: &self.w_ctx + &rhs.w_ctx,
    }
  }
}

impl<'r, R, L> Add<&'r STRState<R>> for STRState<L>
where
  R: ndarray::Data<Elem = f64>,
  L: ndarray::Data<Elem = f64>,
{
  type Output = STRState<OwnedRepr<f64>>;
  fn add(self, rhs: &'r STRState<R>) -> Self::Output {
    &self + rhs
  }
}

impl<'l, R, L> Add<STRState<R>> for &'l STRState<L>
where
  R: ndarray::Data<Elem = f64>,
  L: ndarray::Data<Elem = f64>,
{
  type Output = STRState<OwnedRepr<f64>>;
  fn add(self, rhs: STRState<R>) -> Self::Output {
    self + &rhs
  }
}

impl<R, L> Add<STRState<R>> for STRState<L>
where
  R: ndarray::Data<Elem = f64>,
  L: ndarray::Data<Elem = f64>,
{
  type Output = STRState<OwnedRepr<f64>>;
  fn add(self, rhs: STRState<R>) -> Self::Output {
    &self + &rhs
  }
}

pub fn x_oo<T: ndarray::Data<Elem = f64>>(
  v: &ndarray::ArrayBase<T, Ix1>,
  tht_x: f64,
  sig_x: f64,
) -> ndarray::Array1<f64> {
  1. / (1. + ((tht_x - v) / sig_x).exp())
}

pub fn tau_x<T: ndarray::Data<Elem = f64>>(
  v: &ndarray::ArrayBase<T, Ix1>,
  tau_x_0: f64,
  tau_x_1: f64,
  tht_x_t: f64,
  sig_x_t: f64,
) -> ndarray::Array1<f64> {
  tau_x_0 + tau_x_1 / (1. + ((tht_x_t - v) / sig_x_t).exp())
}

impl STRHistory {
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

    let time =
      (Array1::range(0., num_timesteps as f64, 1.) * output_dt).to_shape((num_timesteps, 1)).unwrap().to_owned();

    polars::prelude::DataFrame::new(vec![
      array2_to_polars_column("time", time.view()),
      array2_to_polars_column("v", self.v.slice(srange)),
      array2_to_polars_column("n", self.n.slice(srange)),
      array2_to_polars_column("h", self.h.slice(srange)),
      array2_to_polars_column("r", self.r.slice(srange)),
      array2_to_polars_column("s", self.s.slice(srange)),
      array2_to_polars_column("i_ext", self.i_ext.slice(erange)),
    ])
    .expect("This shouldn't happend if the struct is valid")
  }
}
