use std::fmt::Debug;
use std::ops::Add;
use std::ops::Mul;

use ndarray::{s, Array1, Array2, ArrayBase, ArrayView1, Ix1, Ix2, OwnedRepr, ViewRepr};
use pyo3::PyAny;
use pyo3::Python;
use struct_field_names_as_array::FieldNamesAsSlice;

use crate::parameters::STNParameters;
use crate::types::*;
use crate::util::*;

#[derive(Default, Debug)]
pub struct STN;

impl Neuron for STN {
  const TYPE: &'static str = "STN";
}

pub type STNConfig = NeuronConfig<STN, STNParameters, STNPopulationBoundryConditions>;

#[derive(FieldNamesAsSlice, Debug)]
pub struct STNPopulationBoundryConditions {
  pub count: usize,
  // State
  pub v: Array1<f64>,
  pub n: Array1<f64>,
  pub h: Array1<f64>,
  pub r: Array1<f64>,
  pub ca: Array1<f64>,
  pub s: Array1<f64>,
  pub i_ext: pyo3::Py<PyAny>,

  // Connection Matrice
  pub w_gpe: Array2<f64>,
  pub w_ctx: Array2<f64>,
}

impl Build<STN, Boundary> for STNPopulationBoundryConditions {
  const PYTHON_CALLABLE_FIELD_NAMES: &[&'static str] = &["i_ext"];
}

pub type BuilderSTNBoundary = Builder<STN, Boundary, STNPopulationBoundryConditions>;

impl Builder<STN, Boundary, STNPopulationBoundryConditions> {
  pub fn finish(self, stn_count: usize, gpe_count: usize, ctx_count: usize) -> STNPopulationBoundryConditions {
    // TODO: Decide what should happen with the defaults?
    // Instead of zero it should panic if top level use_default is set to false
    let zero = Array1::zeros(stn_count);

    let v =
      self.map.get("v").map(try_toml_value_to_1darray::<f64>).map_or(zero.clone(), |x| x.expect("invalid bc dim v"));
    assert_eq!(v.len(), stn_count);
    let n =
      self.map.get("n").map(try_toml_value_to_1darray::<f64>).map_or(zero.clone(), |x| x.expect("invalid bc dim n"));
    assert_eq!(n.len(), stn_count);
    let h =
      self.map.get("h").map(try_toml_value_to_1darray::<f64>).map_or(zero.clone(), |x| x.expect("invalid bc dim h"));
    assert_eq!(h.len(), stn_count);
    let r =
      self.map.get("r").map(try_toml_value_to_1darray::<f64>).map_or(zero.clone(), |x| x.expect("invalid bc dim r"));
    assert_eq!(r.len(), stn_count);
    let ca =
      self.map.get("ca").map(try_toml_value_to_1darray::<f64>).map_or(zero.clone(), |x| x.expect("invalid bc dim ca"));
    assert_eq!(ca.len(), stn_count);
    let s =
      self.map.get("s").map(try_toml_value_to_1darray::<f64>).map_or(zero.clone(), |x| x.expect("invalid bc dim s"));
    assert_eq!(s.len(), stn_count);

    let i_ext = toml_py_function_qualname_to_py_object(self.map.get("i_ext").expect("default should be set by caller"));

    let w_gpe = self
      .map
      .get("w_gpe")
      .map(try_toml_value_to_2darray::<f64>)
      .map_or(Array2::zeros((gpe_count, stn_count)), |x| x.expect("invalid bc for w_gpe"));
    assert_eq!(w_gpe.shape(), &[gpe_count, stn_count]);

    let w_ctx = self
      .map
      .get("w_ctx")
      .map(try_toml_value_to_2darray::<f64>)
      .map_or(Array2::zeros((ctx_count, stn_count)), |x| x.expect("invalid bc for w_ctx"));
    assert_eq!(w_ctx.shape(), &[ctx_count, stn_count]);

    STNPopulationBoundryConditions { count: stn_count, v, n, h, r, ca, s, w_gpe, w_ctx, i_ext }
  }
}

impl STNPopulationBoundryConditions {
  pub fn to_toml(&self, i_ext_py_qualified_name: &str) -> toml::Value {
    let Self { count, v, h, n, r, ca, w_gpe, w_ctx, s, .. } = self;
    let mut table = toml::value::Table::new();
    table.insert("count".to_owned(), (*count as i64).into());
    table.insert("v".to_owned(), v.to_vec().into());
    table.insert("n".to_owned(), n.to_vec().into());
    table.insert("h".to_owned(), h.to_vec().into());
    table.insert("r".to_owned(), r.to_vec().into());
    table.insert("ca".to_owned(), ca.to_vec().into());
    table.insert("s".to_owned(), s.to_vec().into());
    table.insert("w_gpe".to_owned(), w_gpe.rows().into_iter().map(|x| x.to_vec()).collect::<Vec<_>>().into());
    table.insert("w_ctx".to_owned(), w_ctx.rows().into_iter().map(|x| x.to_vec()).collect::<Vec<_>>().into());
    table.insert("i_ext".to_owned(), toml::Value::String(i_ext_py_qualified_name.to_owned()));

    toml::Value::Table(table)
  }
}

pub struct STNHistory {
  pub v: Array2<f64>,
  pub n: Array2<f64>,
  pub h: Array2<f64>,
  pub r: Array2<f64>,
  pub ca: Array2<f64>,
  pub s: Array2<f64>,
  pub w_gpe: Array2<f64>,
  pub ca_g_s: Array2<f64>,
  pub w_ctx: Array2<f64>,
  pub i_ext: Array2<f64>,
  pub i_ext_f: pyo3::Py<PyAny>,
  pub i_gpe: Array2<f64>,
  pub i_ctx: Array2<f64>,
}

// How about: the structure of my noise, not really because i want to enforce a min_isi is
// already known white i want as input is just a function
// of the stimuli being on or off maybe
//
// but still I can just generate it no?
// Say i want a batch of the noise
// STIMULI + SEED --> CTX NOISE
//  TIME INTERVAL /  then W_CTX in every nuclei

impl STNHistory {
  pub fn new(
    num_timesteps: usize,
    stn_count: usize,
    gpe_count: usize,
    ctx_count: usize,
    edge_resolution: usize,
    bc: STNPopulationBoundryConditions,
  ) -> Self {
    let mut res = Self {
      v: Array2::zeros((num_timesteps, stn_count)),
      n: Array2::zeros((num_timesteps, stn_count)),
      h: Array2::zeros((num_timesteps, stn_count)),
      r: Array2::zeros((num_timesteps, stn_count)),
      ca: Array2::zeros((num_timesteps, stn_count)),
      s: Array2::zeros((num_timesteps, stn_count)),
      w_gpe: Array2::zeros((gpe_count, stn_count)),
      ca_g_s: Array2::zeros((gpe_count, stn_count)),
      w_ctx: Array2::zeros((ctx_count, stn_count)),
      i_ext: Array2::zeros((num_timesteps * edge_resolution, stn_count)),
      i_ext_f: bc.i_ext,
      i_gpe: Array2::zeros((num_timesteps, stn_count)),
      i_ctx: Array2::zeros((num_timesteps, stn_count)),
    };

    res.v.row_mut(0).assign(&bc.v);
    res.n.row_mut(0).assign(&bc.n);
    res.h.row_mut(0).assign(&bc.h);
    res.r.row_mut(0).assign(&bc.r);
    res.ca.row_mut(0).assign(&bc.ca);
    res.s.row_mut(0).assign(&bc.s);
    res.w_gpe.assign(&bc.w_gpe);
    res.w_ctx.assign(&bc.w_ctx);

    res
  }

  pub fn fill_i_ext(&mut self, py: Python, start_time: f64, end_time: f64, dt: f64) {
    let neurons = self.i_ext.shape()[1];
    self.i_ext = vectorize_i_ext_py(py, &self.i_ext_f, start_time, end_time, dt, neurons);
  }

  pub fn roll(&mut self) {
    // FIXME check if there is an off by one with this method
    roll_time_series(self.v.view_mut());
    roll_time_series(self.n.view_mut());
    roll_time_series(self.h.view_mut());
    roll_time_series(self.r.view_mut());
    roll_time_series(self.ca.view_mut());
    roll_time_series(self.s.view_mut());
    roll_time_series(self.i_gpe.view_mut());
    roll_time_series(self.i_ctx.view_mut());
  }

  pub fn size(&self) -> usize {
    std::mem::size_of::<f64>()
      * (self.v.len()
        + self.n.len()
        + self.h.len()
        + self.r.len()
        + self.ca.len()
        + self.s.len()
        + self.w_gpe.len()
        + self.ca_g_s.len()
        + self.i_ext.len()
        + self.i_gpe.len()
        + self.i_ctx.len())
  }

  pub fn insert(&mut self, it: usize, y: &STNState<OwnedRepr<f64>>, [i_gpe, i_ctx]: [Array1<f64>; 2]) {
    self.v.row_mut(it).assign(&y.v);
    self.n.row_mut(it).assign(&y.n);
    self.r.row_mut(it).assign(&y.r);
    self.h.row_mut(it).assign(&y.h);
    self.ca.row_mut(it).assign(&y.ca);
    self.s.row_mut(it).assign(&y.s);
    self.i_gpe.row_mut(it).assign(&i_gpe);
    self.i_ctx.row_mut(it).assign(&i_ctx);
    self.w_gpe = y.w_gpe.clone();
    self.w_ctx = y.w_ctx.clone();
    self.ca_g_s = y.ca_gpe.clone();
  }

  pub fn row<'a>(&'a self, it: usize) -> STNState<ViewRepr<&'a f64>> {
    STNState {
      v: self.v.row(it),
      n: self.n.row(it),
      h: self.h.row(it),
      r: self.r.row(it),
      ca: self.ca.row(it),
      s: self.s.row(it),
      w_gpe: self.w_gpe.view(),
      ca_gpe: self.ca_g_s.view(),
      w_ctx: self.w_ctx.view(),
    }
  }
}

impl<'a, T> Mul<&'a STNState<T>> for f64
where
  T: ndarray::Data<Elem = f64>,
{
  type Output = STNState<OwnedRepr<f64>>;
  fn mul(self, rhs: &'a STNState<T>) -> Self::Output {
    STNState {
      v: self * &rhs.v,
      n: self * &rhs.n,
      h: self * &rhs.h,
      r: self * &rhs.r,
      ca: self * &rhs.ca,
      s: self * &rhs.s,
      w_gpe: self * &rhs.w_gpe,
      ca_gpe: self * &rhs.ca_gpe,
      w_ctx: self * &rhs.w_ctx,
    }
  }
}

impl<T> Mul<STNState<T>> for f64
where
  T: ndarray::Data<Elem = f64>,
{
  type Output = STNState<OwnedRepr<f64>>;
  fn mul(self, rhs: STNState<T>) -> Self::Output {
    self * &rhs
  }
}

impl<'a, T> Mul<f64> for &'a STNState<T>
where
  T: ndarray::Data<Elem = f64>,
{
  type Output = STNState<OwnedRepr<f64>>;
  fn mul(self, rhs: f64) -> Self::Output {
    rhs * self
  }
}

impl<'a, T> Mul<f64> for STNState<T>
where
  T: ndarray::Data<Elem = f64>,
{
  type Output = STNState<OwnedRepr<f64>>;
  fn mul(self, rhs: f64) -> Self::Output {
    rhs * &self
  }
}

pub struct STNState<T>
where
  T: ndarray::RawData,
{
  pub v: ArrayBase<T, Ix1>,
  pub n: ArrayBase<T, Ix1>,
  pub h: ArrayBase<T, Ix1>,
  pub r: ArrayBase<T, Ix1>,
  pub ca: ArrayBase<T, Ix1>,
  pub s: ArrayBase<T, Ix1>,
  pub w_gpe: ArrayBase<T, Ix2>,
  pub ca_gpe: ArrayBase<T, Ix2>,
  pub w_ctx: ArrayBase<T, Ix2>,
}

pub struct DiracDeltaState<T>
where
  T: ndarray::Data,
{
  pub d: ArrayBase<T, Ix2>,
  pub dt_spike: ArrayBase<T, Ix2>,
}

impl DiracDeltaState<OwnedRepr<f64>> {
  pub fn new<Sh: ndarray::ShapeBuilder<Dim = Ix2> + Clone>(shape: Sh) -> Self {
    Self { d: Array2::zeros(shape.clone()), dt_spike: Array2::zeros(shape) }
  }
}

impl<T> DiracDeltaState<T>
where
  T: ndarray::DataMut<Elem = f64>,
{
  pub fn update<TS: ndarray::Data<Elem = f64>>(&mut self, v: &ArrayBase<TS, Ix1>, dt: f64, it: usize) {
    for n in 0..self.d.shape()[1] {
      self.dt_spike[[it + 1, n]] = self.dt_spike[[it, n]] + dt;
      if v[n] > 0. && self.dt_spike[[it + 1, n]] > 2. {
        self.d[[it, n]] = 1.;
        self.dt_spike[[it + 1, n]] = 0.;
      } else {
        self.d[[it + 1, n]] = 0.;
      }
    }
  }
}

impl<T> Debug for STNState<T>
where
  T: ndarray::Data<Elem = f64>,
{
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_struct("STNState")
      .field("v", &self.v)
      .field("n", &self.n)
      .field("r", &self.r)
      .field("ca", &self.ca)
      .field("s", &self.s)
      .field("w_gpe", &self.w_gpe)
      .finish()
  }
}

impl<T> STNState<T>
where
  T: ndarray::Data<Elem = f64>,
{
  pub fn dydt(
    &self,
    p: &STNParameters,
    d_stn: &ArrayView1<f64>,
    d_gpe: &ArrayView1<f64>,
    s_gpe: &ArrayView1<f64>,
    s_ctx: &ArrayView1<f64>,
    i_ext: &ArrayView1<f64>,
  ) -> (STNState<OwnedRepr<T::Elem>>, [Array1<f64>; 2]) {
    let Self { v, n, h, r, ca, s, w_gpe, ca_gpe: ca_g_s, w_ctx } = self;
    let eta_d: f64 = 0.01; // @TODO -> Factor this into parameters struct
    let eta_p: f64 = 0.0075;
    let f_d: f64 = 0.42;
    let f_p: f64 = 2.25;

    let _etas = [0., eta_d, eta_p];
    let _fs = [0., f_d, f_p];

    let n_oo = x_oo(v, p.tht_n, p.sig_n);
    let m_oo = x_oo(v, p.tht_m, p.sig_m);
    let h_oo = x_oo(v, p.tht_h, p.sig_h);
    let a_oo = x_oo(v, p.tht_a, p.sig_a);
    let r_oo = x_oo(v, p.tht_r, p.sig_r);
    let s_oo = x_oo(v, p.tht_s, p.sig_s);
    let b_oo = x_oo(r, p.tht_b, -p.sig_b) - p.b_const;
    let h_syn_oo = x_oo(&(v - p.tht_g), p.tht_g_h, p.sig_g_h);

    let tau_n = tau_x(v, p.tau_n_0, p.tau_n_1, p.tht_n_t, p.sig_n_t);
    let tau_h = tau_x(v, p.tau_h_0, p.tau_h_1, p.tht_h_t, p.sig_h_t);
    let tau_r = tau_x(v, p.tau_r_0, p.tau_r_1, p.tht_r_t, p.sig_r_t);

    let i_l = p.g_l * (v - p.v_l);
    let i_k = p.g_k * n.powi(4) * (v - p.v_k);
    let i_na = p.g_na * m_oo.powi(3) * h * (v - p.v_na);
    let i_t = p.g_t * a_oo.powi(3) * b_oo.pow2() * (v - p.v_ca);
    let i_ca = p.g_ca * s_oo.powi(2) * (v - p.v_ca);
    let i_ahp = p.g_ahp * (v - p.v_k) * ca / (ca + p.k_1);
    let i_gpe = p.g_g_s * (v - p.v_g_s) * (self.w_gpe.t().dot(s_gpe));
    let i_ctx = p.g_ctx * (v - p.v_ctx) * (w_ctx.t().dot(s_ctx));

    //   let mut dw_gpe = Array2::<f64>::zeros(w_gpe.raw_dim());
    //   let mut _ca_g_s = Array2::<f64>::zeros(ca_g_s.raw_dim());
    //   for &(i, j) in a_g_s.iter() {
    //     _ca_g_s[[i, j]] = -ca_g_s[[i, j]] / p.tau_ca + p.ca_pre * d_gpe[i] + p.ca_post * d_stn[j];
    //     let k = (ca_g_s[[i, j]] > p.theta_d) as usize + (ca_g_s[[i, j]] > p.theta_p) as usize;
    //     dw_gpe[[i, j]] = etas[k] * (fs[k] - w_gpe[[i, j]]);
    //   }

    let dy = STNState {
      v: -i_l - i_k - i_na - &i_t - &i_ca - i_ahp - &i_gpe - &i_ctx - i_ext,
      n: p.phi_n * (n_oo - n) / tau_n,
      h: p.phi_h * (h_oo - h) / tau_h,
      r: p.phi_r * (r_oo - r) / tau_r,
      ca: p.eps * ((-i_ca - i_t) - p.k_ca * ca),
      s: p.alpha * h_syn_oo * (1. - s) - p.beta * s,
      ca_gpe: -ca_g_s / p.tau_ca + p.ca_pre * &d_gpe.iax(1) + p.ca_post * &d_stn.iax(0),
      w_gpe: Array2::<f64>::zeros(w_gpe.raw_dim()),
      w_ctx: Array2::<f64>::zeros(w_ctx.raw_dim()),
    };

    (dy, [i_gpe, i_ctx])
  }
}

impl<'l, 'r, R, L> Add<&'r STNState<R>> for &'l STNState<L>
where
  R: ndarray::Data<Elem = f64>,
  L: ndarray::Data<Elem = f64>,
{
  type Output = STNState<OwnedRepr<f64>>;
  fn add(self, rhs: &'r STNState<R>) -> Self::Output {
    STNState {
      v: &self.v + &rhs.v,
      n: &self.n + &rhs.n,
      h: &self.h + &rhs.h,
      r: &self.r + &rhs.r,
      ca: &self.ca + &rhs.ca,
      s: &self.s + &rhs.s,
      w_gpe: &self.w_gpe + &rhs.w_gpe,
      ca_gpe: &self.w_gpe + &rhs.w_gpe,
      w_ctx: &self.w_ctx + &rhs.w_ctx,
    }
  }
}

impl<'r, R, L> Add<&'r STNState<R>> for STNState<L>
where
  R: ndarray::Data<Elem = f64>,
  L: ndarray::Data<Elem = f64>,
{
  type Output = STNState<OwnedRepr<f64>>;
  fn add(self, rhs: &'r STNState<R>) -> Self::Output {
    &self + rhs
  }
}

impl<'l, R, L> Add<STNState<R>> for &'l STNState<L>
where
  R: ndarray::Data<Elem = f64>,
  L: ndarray::Data<Elem = f64>,
{
  type Output = STNState<OwnedRepr<f64>>;
  fn add(self, rhs: STNState<R>) -> Self::Output {
    self + &rhs
  }
}

impl<R, L> Add<STNState<R>> for STNState<L>
where
  R: ndarray::Data<Elem = f64>,
  L: ndarray::Data<Elem = f64>,
{
  type Output = STNState<OwnedRepr<f64>>;
  fn add(self, rhs: STNState<R>) -> Self::Output {
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

impl STNHistory {
  #[rustfmt::skip]
  pub fn into_compressed_polars_df(
    &self,
    idt: f64,
    odt: Option<f64>,
    edge_resolution: usize,
    data: &Vec<&str>,
    start_time: f64,
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

    let time = start_time +
      (Array1::range(0., num_timesteps as f64, 1.) * output_dt).to_shape((num_timesteps, 1)).unwrap().to_owned() ;

    let mut out = vec![];
    if data.contains(&"time")  { out.push(array2_to_polars_column("time",  time      .view())) }
    if data.contains(&"v")     { out.push(array2_to_polars_column("v",     self.v    .slice(srange))) }
    if data.contains(&"n")     { out.push(array2_to_polars_column("n",     self.n    .slice(srange))) }
    if data.contains(&"h")     { out.push(array2_to_polars_column("h",     self.h    .slice(srange))) }
    if data.contains(&"r")     { out.push(array2_to_polars_column("r",     self.r    .slice(srange))) }
    if data.contains(&"ca")    { out.push(array2_to_polars_column("ca",    self.ca   .slice(srange))) }
    if data.contains(&"s")     { out.push(array2_to_polars_column("s",     self.s    .slice(srange))) }
    if data.contains(&"i_ext") { out.push(array2_to_polars_column("i_ext", self.i_ext.slice(erange))) }
    if data.contains(&"i_gpe") { out.push(array2_to_polars_column("i_gpe", self.i_gpe.slice(srange))) }
    if data.contains(&"i_ctx") { out.push(array2_to_polars_column("i_ctx", self.i_ctx.slice(srange))) }
    if data.contains(&"w_gpe") { out.push(  unit_to_polars_column("w_gpe", self.w_gpe.view(), num_timesteps)) }

    polars::prelude::DataFrame::new(out)
    .expect("This shouldn't happend if the struct is valid")
  }
}
