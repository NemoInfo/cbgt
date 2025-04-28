use log::debug;
use ndarray::{s, Array1, Array2, ArrayView1};
use struct_field_names_as_array::FieldNamesAsSlice;

use crate::parameters::GPeParameters;
use crate::types::*;
use crate::util::*;

#[derive(Default)]
pub struct GPe;

impl Neuron for GPe {
  const TYPE: &'static str = "GPe";
}

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
}

impl Build<GPe, Boundary> for GPePopulationBoundryConditions {
  const PYTHON_CALLABLE_FIELD_NAMES: &[&'static str] = &["i_ext", "i_app"];
}

pub type BuilderGPeBoundary = Builder<GPe, Boundary, GPePopulationBoundryConditions>;

impl BuilderGPeBoundary {
  pub fn finish(self, gpe_count: usize, stn_count: usize, dt: f64, total_t: f64) -> GPePopulationBoundryConditions {
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
    let i_ext = vectorize_i_ext_py(&i_ext_f, dt, total_t, stn_count);

    let i_app_f =
      toml_py_function_qualname_to_py_object(self.map.get("i_app").expect("default should be set by caller"));
    let i_app = vectorize_i_ext_py(&i_app_f, dt, total_t, stn_count);

    debug!("GPe I_ext vectorized to\n{i_ext}");
    debug!("GPe I_app vectorized to\n{i_app}");

    let w_g_g = self
      .map
      .get("w_g_g")
      .map(try_toml_value_to_2darray::<f64>)
      .map_or(Array2::zeros((gpe_count, stn_count)), |x| x.expect("invalid bc for w_g_g"));
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
      .map_or(Array2::zeros((gpe_count, stn_count)), |x| x.expect("invalid bc for w_s_g"));
    let c_s_g = self
      .map
      .get("c_s_g")
      .map(try_toml_value_to_2darray::<f64>)
      .map_or(w_s_g.mapv(|x| (x != 0.) as u8 as f64), |x| x.expect("invalid bc for c_s_g"))
      .mapv(|x| (x != 0.) as u8 as f64);
    assert_eq!(c_s_g.shape(), &[stn_count, gpe_count]);

    GPePopulationBoundryConditions { count: gpe_count, v, n, h, r, ca, s, c_g_g, w_g_g, c_s_g, w_s_g, i_ext, i_app }
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
  pub w_g_g: Array2<f64>,
  pub c_g_g: Array2<f64>,
  pub w_s_g: Array2<f64>,
  pub c_s_g: Array2<f64>,
  // TODO

  // Connection Currents
  pub i_g_g: Array2<f64>,
  pub i_s_g: Array2<f64>,

  // STDP
  pub rho_pre: Array2<f64>,
  pub rho_post: Array2<f64>,
  pub dt_spike: Array1<f64>,
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
    self.w_g_g.assign(&bc.w_g_g);
    self.w_s_g.assign(&bc.w_s_g);
    self.i_ext.assign(&bc.i_ext);
    self.i_app.assign(&bc.i_app);
    self
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
      w_s_g: Array2::zeros((stn_count, gpe_count)),
      w_g_g: Array2::zeros((gpe_count, gpe_count)),
      c_s_g: Array2::zeros((stn_count, gpe_count)),
      c_g_g: Array2::zeros((gpe_count, gpe_count)),
      rho_pre: Array2::zeros((num_timesteps, gpe_count)),
      rho_post: Array2::zeros((num_timesteps, gpe_count)),
      dt_spike: Array1::from_elem(gpe_count, f64::INFINITY),
    }
  }

  pub fn euler_step(
    &mut self,
    it: usize,
    dt: f64,
    p: &GPeParameters,
    s_stn: &ArrayView1<f64>,
    rho_pre_stn: &ArrayView1<f64>,
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

    let tau_n = &tau_x(v, p.tau_n_0, p.tau_n_1, p.tht_n_t, p.sig_n_t);
    let tau_h = &tau_x(v, p.tau_h_0, p.tau_h_1, p.tht_h_t, p.sig_h_t);

    // Compute currents
    let mut i_l = self.i_l.row_mut(it);
    let mut i_k = self.i_k.slice_mut(t);
    let mut i_na = self.i_na.slice_mut(t);
    let mut i_t = self.i_t.slice_mut(t);
    let mut i_ca = self.i_ca.slice_mut(t);
    let mut i_ahp = self.i_ahp.slice_mut(t);
    let mut i_s_g = self.i_s_g.slice_mut(t);
    let mut i_g_g = self.i_g_g.slice_mut(t);

    i_l.assign(&(p.g_l * (v - p.v_l)));
    i_k.assign(&(p.g_k * n.powi(4) * (v - p.v_k)));
    i_na.assign(&(p.g_na * m_inf.powi(3) * h * (v - p.v_na)));
    i_t.assign(&(p.g_t * a_inf.powi(3) * r * (v - p.v_ca)));
    i_ca.assign(&(p.g_ca * s_inf.powi(2) * (v - p.v_ca)));
    i_ahp.assign(&(p.g_ahp * (v - p.v_k) * ca / (ca + p.k_1)));
    i_s_g.assign(&(p.g_s_g * (v - p.v_s_g) * (self.w_s_g.t().dot(s_stn))));
    i_g_g.assign(&(p.g_g_g * (v - p.v_g_g) * (self.w_g_g.t().dot(s))));

    // Update state
    v1.assign(
      &(v
        + dt
          * (-&i_l - &i_k - &i_na - &i_t - &i_ca - &i_ahp - &i_s_g - &i_g_g - &self.i_ext.row(it)
            + &self.i_app.row(it))),
    );
    n1.assign(&(n + dt * p.phi_n * (n_inf - n) / tau_n));
    h1.assign(&(h + dt * p.phi_h * (h_inf - h) / tau_h));
    r1.assign(&(r + dt * p.phi_r * (r_inf - r) / p.tau_r));
    ca1.assign(&(ca + dt * p.eps * ((-&i_ca - &i_t) - p.k_ca * ca)));

    self.dt_spike += dt;
    self.dt_spike.zip_mut_with(v, |dspike, &v| {
      if *dspike > p.min_dt_spike && v > p.tht_spike {
        *dspike = 0.;
      }
    });
    ndarray::Zip::from(&mut rho_pre1).and(rho_pre).and(&self.dt_spike).for_each(|rho1, &rho, &dt_spike| {
      *rho1 = if dt_spike == 0. { 1. } else { rho - dt / p.tau_pre * rho };
    });
    ndarray::Zip::from(&mut rho_post1).and(rho_post).and(&self.dt_spike).for_each(|rho1, &rho, &dt_spike| {
      *rho1 = if dt_spike == 0. { 1. } else { rho - dt / p.tau_post * rho };
    });

    // Update synapses
    let h_syn_inf = x_inf(&(v - p.tht_g).view(), p.tht_g_h, p.sig_g_h);
    s1.assign(&(s + dt * (p.alpha * h_syn_inf * (1. - s) - p.beta * s)));

    let de = (p.a_pre * rho_pre_stn.iax(1).dot(&rho_post.iax(0).mapv(|x| (x == 1.).as_f64()))
      - p.a_post * rho_post.iax(1).dot(&rho_pre_stn.iax(0).mapv(|x| (x == 1.).as_f64())))
      * &self.c_s_g;
    self.w_s_g += &de;
    self.w_s_g.mapv_inplace(|x| x.max(0.).min(5.));

    let de = (p.a_pre * rho_pre.iax(1).dot(&rho_post.iax(0).mapv(|x| (x == 1.).as_f64()))
      - p.a_post * rho_post.iax(1).dot(&rho_pre.iax(0).mapv(|x| (x == 1.).as_f64())))
      * &self.c_g_g;

    self.w_g_g += &de;
    self.w_g_g.mapv_inplace(|x| x.max(0.).min(5.));
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
      array2_to_polars_column("i_g_g", self.i_g_g.slice(srange)),
      array2_to_polars_column("i_s_g", self.i_s_g.slice(srange)),
      array2_to_polars_column("i_ext", self.i_ext.slice(srange)),
      array2_to_polars_column("s", self.s.slice(srange)),
      unit_to_polars_column("w_g_g", self.w_g_g.view(), num_timesteps),
    ])
    .expect("This shouldn't happend if the struct is valid")
  }
}
