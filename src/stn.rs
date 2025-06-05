use std::fmt::Debug;
use std::ops::Add;
use std::ops::Mul;

use log::debug;
use ndarray::azip;
use ndarray::Array3;
use ndarray::{s, Array1, Array2, ArrayView1, Zip};
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

#[derive(FieldNamesAsSlice, Debug, Default)]
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
  pub w_g_s: Array2<f64>,
  pub c_g_s: Array2<f64>,
}

impl Build<STN, Boundary> for STNPopulationBoundryConditions {
  const PYTHON_CALLABLE_FIELD_NAMES: &[&'static str] = &["i_ext"];
}

pub type BuilderSTNBoundary = Builder<STN, Boundary, STNPopulationBoundryConditions>;

impl Builder<STN, Boundary, STNPopulationBoundryConditions> {
  pub fn finish(
    self,
    stn_count: usize,
    gpe_count: usize,
    dt: f64,
    total_t: f64,
    edge_resolution: u8,
  ) -> STNPopulationBoundryConditions {
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

    let i_ext_f =
      toml_py_function_qualname_to_py_object(self.map.get("i_ext").expect("default should be set by caller"));
    let i_ext = vectorize_i_ext_py(&i_ext_f, dt / (edge_resolution as f64), total_t, stn_count);
    assert_eq!(i_ext.shape()[1], stn_count);
    debug!("STN I_ext vectorized to\n{i_ext}");

    let w_g_s = self
      .map
      .get("w_g_s")
      .map(try_toml_value_to_2darray::<f64>)
      .map_or(Array2::zeros((gpe_count, stn_count)), |x| x.expect("invalid bc for w_g_s"));
    let c_g_s = self
      .map
      .get("c_g_s")
      .map(try_toml_value_to_2darray::<f64>)
      .map_or(w_g_s.mapv(|x| (x != 0.) as u8 as f64), |x| x.expect("invalid bc for c_g_s"))
      .mapv(|x| (x != 0.) as u8 as f64);
    assert_eq!(w_g_s.shape(), &[gpe_count, stn_count]);

    STNPopulationBoundryConditions { count: stn_count, v, n, h, r, ca, s, w_g_s, c_g_s, i_ext }
  }
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
    table.insert("w_g_s".to_owned(), self.w_g_s.rows().into_iter().map(|x| x.to_vec()).collect::<Vec<_>>().into());
    table.insert("i_ext".to_owned(), toml::Value::String(i_ext_py_qualified_name.to_owned()));

    toml::Value::Table(table)
  }
}

#[derive(Clone)]
pub struct STNPopulation {
  // TODO add count here, why tho?
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

  // Connection Matrices
  pub w_g_s: Array2<f64>,
  pub c_g_s: Array2<f64>,

  // Connection Currents
  pub i_g_s: Array2<f64>,

  // STDP
  pub rho_pre: Array2<f64>,
  pub rho_post: Array2<f64>,
  pub dt_spike: Array1<f64>,
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
    self.w_g_s.assign(&bc.w_g_s);
    self.i_ext.assign(&bc.i_ext);
    self
  }

  pub fn new(num_timesteps: usize, stn_count: usize, gpe_count: usize, edge_resolution: usize) -> Self {
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
      c_g_s: Array2::zeros((gpe_count, stn_count)),
      w_g_s: Array2::zeros((gpe_count, stn_count)),
      rho_pre: Array2::zeros((num_timesteps, stn_count)),
      rho_post: Array2::zeros((num_timesteps, stn_count)),
      dt_spike: Array1::from_elem(stn_count, f64::INFINITY),

      i_ext: Array2::zeros((num_timesteps * edge_resolution, stn_count)),
    }
  }

  pub fn euler_step(
    &mut self,
    it: usize,
    dt: f64,
    p: &STNParameters,
    s_gpe: &ArrayView1<f64>,
    rho_pre_gpe: &ArrayView1<f64>,
  ) {
    let t = s![it, ..];
    let t1 = s![it + 1, ..];

    let (v, mut v1) = self.v.multi_slice_mut((t, t1));
    let (r, mut r1) = self.r.multi_slice_mut((t, t1));
    let (n, mut n1) = self.n.multi_slice_mut((t, t1));
    let (h, mut h1) = self.h.multi_slice_mut((t, t1));
    let (ca, mut ca1) = self.ca.multi_slice_mut((t, t1));
    let (s, mut s1) = self.s.multi_slice_mut((t, t1));
    let (rho_pre, mut rho_pre1) = self.rho_pre.multi_slice_mut((t, t1));
    let (rho_post, mut rho_post1) = self.rho_post.multi_slice_mut((t, t1));

    let (v, r, n, h, ca, s, rho_pre, rho_post) =
      (&v.view(), &r.view(), &n.view(), &h.view(), &ca.view(), &s.view(), &rho_pre.view(), &rho_post.view());

    let n_inf = &x_inf(v, p.tht_n, p.sig_n);
    let m_inf = &x_inf(v, p.tht_m, p.sig_m);
    let h_inf = &x_inf(v, p.tht_h, p.sig_h);
    let a_inf = &x_inf(v, p.tht_a, p.sig_a);
    let r_inf = &x_inf(v, p.tht_r, p.sig_r);
    let s_inf = &x_inf(v, p.tht_s, p.sig_s);
    let b_inf = &x_inf(r, p.tht_b, -p.sig_b) - p.b_const; // [!]

    let tau_n = &tau_x(v, p.tau_n_0, p.tau_n_1, p.tht_n_t, p.sig_n_t);
    let tau_h = &tau_x(v, p.tau_h_0, p.tau_h_1, p.tht_h_t, p.sig_h_t);
    let tau_r = &tau_x(v, p.tau_r_0, p.tau_r_1, p.tht_r_t, p.sig_r_t);

    // Compute currents
    let mut i_l = self.i_l.slice_mut(t);
    let mut i_k = self.i_k.slice_mut(t);
    let mut i_na = self.i_na.slice_mut(t);
    let mut i_t = self.i_t.slice_mut(t);
    let mut i_ca = self.i_ca.slice_mut(t);
    let mut i_ahp = self.i_ahp.slice_mut(t);
    let mut i_g_s = self.i_g_s.slice_mut(t);

    i_l.assign(&(p.g_l * (v - p.v_l)));
    i_k.assign(&(p.g_k * n.powi(4) * (v - p.v_k)));
    i_na.assign(&(p.g_na * m_inf.powi(3) * h * (v - p.v_na)));
    i_t.assign(&(p.g_t * a_inf.powi(3) * b_inf.pow2() * (v - p.v_ca)));
    i_ca.assign(&(p.g_ca * s_inf.powi(2) * (v - p.v_ca)));
    i_ahp.assign(&(p.g_ahp * (v - p.v_k) * ca / (ca + p.k_1)));
    i_g_s.assign(&(p.g_g_s * (v - p.v_g_s) * (self.w_g_s.t().dot(s_gpe))));

    // Update state
    let dv = -&i_l - &i_k - &i_na - &i_t - &i_ca - &i_ahp - &i_g_s - &self.i_ext.row(it);

    v1.assign(&(v + dt * dv));
    n1.assign(&(n + dt * p.phi_n * (n_inf - n) / tau_n));
    h1.assign(&(h + dt * p.phi_h * (h_inf - h) / tau_h));
    r1.assign(&(r + dt * p.phi_r * (r_inf - r) / tau_r));
    ca1.assign(&(ca + dt * p.eps * ((-&i_ca - &i_t) - p.k_ca * ca)));

    self.dt_spike += dt;
    self.dt_spike.zip_mut_with(v, |dspike, &v| {
      if *dspike > p.min_dt_spike && v > p.tht_spike {
        *dspike = 0.;
      }
    });
    Zip::from(&mut rho_pre1).and(rho_pre).and(&self.dt_spike).for_each(|rho1, &rho, &dt_spike| {
      *rho1 = if dt_spike == 0. { 1. } else { rho - dt / p.tau_pre * rho };
    });
    Zip::from(&mut rho_post1).and(rho_post).and(&self.dt_spike).for_each(|rho1, &rho, &dt_spike| {
      *rho1 = if dt_spike == 0. { 1. } else { rho - dt / p.tau_post * rho };
    });

    // Update synapses
    let h_syn_inf = x_inf(&(v - p.tht_g).view(), p.tht_g_h, p.sig_g_h);
    s1.assign(&(s + dt * (p.alpha * h_syn_inf * (1. - s) - p.beta * s)));

    let de = (p.a_pre * rho_pre_gpe.iax(1).dot(&rho_post.iax(0).mapv(|x| (x == 1.).as_f64()))
      - p.a_post * rho_post.iax(1).dot(&rho_pre_gpe.iax(0).mapv(|x| (x == 1.).as_f64())))
      * &self.c_g_s;
    self.w_g_s += &de;
    self.w_g_s.mapv_inplace(|x| x.max(0.).min(5.));
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

    let time =
      (Array1::range(0., num_timesteps as f64, 1.) * output_dt).to_shape((num_timesteps, 1)).unwrap().to_owned();

    polars::prelude::DataFrame::new(vec![
      array2_to_polars_column("time", time.view()),
      array2_to_polars_column("v", self.v.slice(srange)),
      array2_to_polars_column("n", self.n.slice(srange)),
      array2_to_polars_column("h", self.h.slice(srange)),
      array2_to_polars_column("r", self.r.slice(srange)),
      array2_to_polars_column("ca", self.ca.slice(srange)),
      array2_to_polars_column("rho_pre", self.rho_pre.slice(srange)),
      array2_to_polars_column("rho_post", self.rho_post.slice(srange)),
      array2_to_polars_column("s", self.s.slice(srange)),
      array2_to_polars_column("i_l", self.i_l.slice(srange)),
      array2_to_polars_column("i_k", self.i_k.slice(srange)),
      array2_to_polars_column("i_na", self.i_na.slice(srange)),
      array2_to_polars_column("i_t", self.i_t.slice(srange)),
      array2_to_polars_column("i_ca", self.i_ca.slice(srange)),
      array2_to_polars_column("i_ahp", self.i_ahp.slice(srange)),
      array2_to_polars_column("i_g_s", self.i_g_s.slice(srange)),
      array2_to_polars_column("i_ext", self.i_ext.slice(srange)),
    ])
    .expect("This shouldn't happend if the struct is valid")
  }
}

use ndarray::{ArrayBase, Ix1, Ix2, OwnedRepr, ViewRepr};

#[derive(Default, Clone)]
pub struct STNHistory {
  pub v: Array2<f64>,
  pub n: Array2<f64>,
  pub h: Array2<f64>,
  pub r: Array2<f64>,
  pub ca: Array2<f64>,
  pub s: Array2<f64>,
  pub w_g_s: Array2<f64>,
  pub ca_g_s: Array3<f64>,
  pub i_ext: Array2<f64>,
}

impl STNHistory {
  pub fn new(num_timesteps: usize, stn_count: usize, gpe_count: usize, edge_resolution: usize) -> Self {
    Self {
      v: Array2::zeros((num_timesteps, stn_count)),
      n: Array2::zeros((num_timesteps, stn_count)),
      h: Array2::zeros((num_timesteps, stn_count)),
      r: Array2::zeros((num_timesteps, stn_count)),
      ca: Array2::zeros((num_timesteps, stn_count)),
      s: Array2::zeros((num_timesteps, stn_count)),
      w_g_s: Array2::zeros((gpe_count, stn_count)),
      ca_g_s: Array3::zeros((num_timesteps * edge_resolution, gpe_count, stn_count)),
      i_ext: Array2::zeros((num_timesteps * edge_resolution, stn_count)),
    }
  }

  pub fn with_bcs(mut self, bc: STNPopulationBoundryConditions) -> Self {
    self.v.row_mut(0).assign(&bc.v);
    self.n.row_mut(0).assign(&bc.n);
    self.h.row_mut(0).assign(&bc.h);
    self.r.row_mut(0).assign(&bc.r);
    self.ca.row_mut(0).assign(&bc.ca);
    self.s.row_mut(0).assign(&bc.s);
    self.w_g_s.assign(&bc.w_g_s);
    self.i_ext.assign(&bc.i_ext);
    self
  }

  pub fn insert(&mut self, it: usize, y: &STNState<OwnedRepr<f64>>) {
    self.v.row_mut(it).assign(&y.v);
    self.n.row_mut(it).assign(&y.n);
    self.r.row_mut(it).assign(&y.r);
    self.h.row_mut(it).assign(&y.h);
    self.ca.row_mut(it).assign(&y.ca);
    self.s.row_mut(it).assign(&y.s);
    self.w_g_s = y.w_g_s.clone();
    self.ca_g_s.slice_mut(s![it, .., ..]).assign(&y.ca_g_s);
  }

  pub fn row<'a>(&'a self, it: usize) -> STNState<ViewRepr<&'a f64>> {
    STNState {
      v: self.v.row(it),
      n: self.n.row(it),
      h: self.h.row(it),
      r: self.r.row(it),
      ca: self.ca.row(it),
      s: self.s.row(it),
      w_g_s: self.w_g_s.view(),
      ca_g_s: self.ca_g_s.slice(s![it, .., ..]),
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
      w_g_s: self * &rhs.w_g_s,
      ca_g_s: self * &rhs.ca_g_s,
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
  pub w_g_s: ArrayBase<T, Ix2>,
  pub ca_g_s: ArrayBase<T, Ix2>,
}

pub struct DiracDeltaState<T>
where
  T: ndarray::Data,
{
  pub d: ArrayBase<T, Ix1>,
  pub dt_spike: ArrayBase<T, Ix1>,
}

impl DiracDeltaState<OwnedRepr<f64>> {
  pub fn new<Sh: ndarray::ShapeBuilder<Dim = Ix1> + Clone>(shape: Sh) -> Self {
    Self { d: Array1::zeros(shape.clone()), dt_spike: Array1::zeros(shape) }
  }
}

impl<T> DiracDeltaState<T>
where
  T: ndarray::DataMut<Elem = f64>,
{
  pub fn update<TS: ndarray::Data<Elem = f64>>(&mut self, v: &ArrayBase<TS, Ix1>, dt: f64) {
    Zip::from(&mut self.d).and(&mut self.dt_spike).and(v).for_each(|d, dt_spike, &v| {
      *dt_spike += dt;
      if v > 0. && *dt_spike > 2. {
        *d = 1.;
        *dt_spike = 0.;
      } else {
        *d = 0.;
      }
    });
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
      .field("w_g_s", &self.w_g_s)
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
    i_ext: &ArrayView1<f64>,
  ) -> STNState<OwnedRepr<T::Elem>> {
    let Self { v, n, h, r, ca, s, w_g_s, ca_g_s } = self;
    let theta_d: f64 = 0.06;
    let theta_p: f64 = 0.09;
    let eta_d: f64 = 0.001;
    let eta_p: f64 = 0.00075;
    let f_d: f64 = 0.42;
    let f_p: f64 = 2.25;

    let etas = [0., eta_d, eta_p];
    let fs = [0., f_d, f_p];

    let mut dw_g_s = Array2::<f64>::zeros(w_g_s.raw_dim());
    azip!((dw in &mut dw_g_s, w in w_g_s, &ca in ca_g_s) {
      let i = (ca > theta_d) as usize + (ca > theta_p) as usize;
      *dw = etas[i] * (fs[i] - w);
    });

    let n_oo = x_oo(v, p.tht_n, p.sig_n);
    let m_oo = x_oo(v, p.tht_m, p.sig_m);
    let h_oo = x_oo(v, p.tht_h, p.sig_h);
    let a_oo = x_oo(v, p.tht_a, p.sig_a);
    let r_oo = x_oo(v, p.tht_r, p.sig_r);
    let s_oo = x_oo(v, p.tht_s, p.sig_s);
    let b_oo = x_oo(r, p.tht_b, -p.sig_b) - p.b_const;
    let h_syn_oo = x_oo(&(v - p.tht_g), p.tht_g_h, p.sig_g_h);

    let tau_n = _tau_x(v, p.tau_n_0, p.tau_n_1, p.tht_n_t, p.sig_n_t);
    let tau_h = _tau_x(v, p.tau_h_0, p.tau_h_1, p.tht_h_t, p.sig_h_t);
    let tau_r = _tau_x(v, p.tau_r_0, p.tau_r_1, p.tht_r_t, p.sig_r_t);

    let i_l = p.g_l * (v - p.v_l);
    let i_k = p.g_k * n.powi(4) * (v - p.v_k);
    let i_na = p.g_na * m_oo.powi(3) * h * (v - p.v_na);
    let i_t = p.g_t * a_oo.powi(3) * b_oo.pow2() * (v - p.v_ca);
    let i_ca = p.g_ca * s_oo.powi(2) * (v - p.v_ca);
    let i_ahp = p.g_ahp * (v - p.v_k) * ca / (ca + p.k_1);
    let i_g_s = p.g_g_s * (v - p.v_g_s) * (self.w_g_s.t().dot(s_gpe));

    let dy = STNState {
      v: -i_l - i_k - i_na - &i_t - &i_ca - i_ahp - i_g_s - i_ext,
      n: p.phi_n * (n_oo - n) / tau_n,
      h: p.phi_h * (h_oo - h) / tau_h,
      r: p.phi_r * (r_oo - r) / tau_r,
      ca: p.eps * ((-i_ca - i_t) - p.k_ca * ca),
      s: p.alpha * h_syn_oo * (1. - s) - p.beta * s,
      w_g_s: dw_g_s,
      ca_g_s: -ca_g_s / p.tau_ca
        + p.ca_pre * &d_gpe.insert_axis(ndarray::Axis(1))
        + p.ca_post * &d_stn.insert_axis(ndarray::Axis(0)),
    };

    dy
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
      w_g_s: &self.w_g_s + &rhs.w_g_s,
      ca_g_s: &self.ca_g_s + &rhs.ca_g_s,
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

pub fn _tau_x<T: ndarray::Data<Elem = f64>>(
  v: &ndarray::ArrayBase<T, Ix1>,
  tau_x_0: f64,
  tau_x_1: f64,
  tht_x_t: f64,
  sig_x_t: f64,
) -> ndarray::Array1<f64> {
  tau_x_0 + tau_x_1 / (1. + ((tht_x_t - v) / sig_x_t).exp())
}

impl STNHistory {
  pub fn from_population(pop: STNPopulation) -> Self {
    let STNPopulation { v, n, h, r, ca, s, w_g_s, i_ext, .. } = pop;
    Self { n, h, r, ca, s, i_ext, ca_g_s: Array3::zeros([v.shape()[0], w_g_s.shape()[0], w_g_s.shape()[1]]), v, w_g_s }
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
    let carange = s![0..num_timesteps;step, .., ..];
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
      array2_to_polars_column("ca", self.ca.slice(srange)),
      array2_to_polars_column("s", self.s.slice(srange)),
      array2_to_polars_column("i_ext", self.i_ext.slice(erange)),
      array3_to_polars_column("ca_g_s", self.ca_g_s.slice(carange)),
    ])
    .expect("This shouldn't happend if the struct is valid")
  }
}
