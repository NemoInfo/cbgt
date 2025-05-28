use std::collections::HashMap;
use std::str::FromStr;
use std::sync::Arc;

use env_logger::fmt::Formatter;
use log::debug;
use ndarray::{Array1, ArrayView1};
use pyo3::{prelude::*, types::PyDict};
use std::io::Write;

use crate::gpe::{BuilderGPeBoundary, GPePopulation};
use crate::stn::{BuilderSTNBoundary, STNPopulation};
use crate::{
  format_number, write_boundary_file, write_parameter_file, write_temp_pyf_file, Boundary, BuilderGPeParameters,
  BuilderSTNParameters, GPeConfig, NeuronData, STNConfig, SpinBarrier, PYF_FILE_NAME,
};
use crate::{gpe::GPeHistory, GPeParameters};
use crate::{stn::STNHistory, STNParameters};

#[pyclass]
pub struct Network {
  #[pyo3(get)]
  pub dt: f64,
  #[pyo3(get)]
  pub total_t: f64,
  pub stn_states: STNHistory,
  pub stn_parameters: STNParameters,
  pub gpe_states: GPeHistory,
  pub gpe_parameters: GPeParameters,
}

impl Network {
  fn run_euler(&mut self) {
    let num_timesteps: usize = (self.total_t / self.dt) as usize;

    debug!("Computing {} timesteps", format_number(num_timesteps));

    let stn_p = self.stn_parameters.clone();
    let gpe_p = self.gpe_parameters.clone();

    let foo = Array1::zeros(10);
    let rho_pre_gpe: &ArrayView1<f64> = &foo.view();
    let rho_pre_stn: &ArrayView1<f64> = &foo.view();

    let spin_barrier = Arc::new(SpinBarrier::new(2));

    let stn = &mut self.stn_states;
    let gpe = &mut self.gpe_states;
    let dt = self.dt;

    let mut s_gpe_shared = gpe.s.row(0).to_owned();
    let s_gpe_ref = unsafe { ndarray::ArrayView1::from_shape_ptr(s_gpe_shared.len(), s_gpe_shared.as_ptr()) };
    let mut s_gpe_mut =
      unsafe { ndarray::ArrayViewMut1::from_shape_ptr(s_gpe_shared.len(), s_gpe_shared.as_mut_ptr()) };
    let mut s_stn_shared = stn.s.row(0).to_owned();
    let s_stn_ref = unsafe { ndarray::ArrayView1::from_shape_ptr(s_stn_shared.len(), s_stn_shared.as_ptr()) };
    let mut s_stn_mut =
      unsafe { ndarray::ArrayViewMut1::from_shape_ptr(s_stn_shared.len(), s_stn_shared.as_mut_ptr()) };

    let barrier1 = spin_barrier.clone();
    let barrier2 = spin_barrier;
    _ = crossbeam::thread::scope(|s| {
      let handle1 = s.spawn(move |_| {
        for it in 0..num_timesteps - 1 {
          let edge_it = it * 2;
          let yp = &stn.row(it);
          barrier1.sym_sync_call(|| s_stn_mut.assign(&yp.s));
          let dy = yp.dydt(&stn_p, &s_gpe_ref, &rho_pre_gpe, &stn.i_ext.row(edge_it));
          let yn = yp + dy * dt;
          stn.insert(it + 1, &yn);
        }
      });

      let handle2 = s.spawn(move |_| {
        for it in 0..num_timesteps - 1 {
          let edge_it = it * 2;
          let yp = &gpe.row(it);
          barrier2.sym_sync_call(|| s_gpe_mut.assign(&yp.s));
          let dy = yp.dydt(&gpe_p, &s_stn_ref, &rho_pre_stn, &gpe.i_ext.row(edge_it), &gpe.i_app.row(edge_it));
          let yn = yp + dy * dt;
          gpe.insert(it + 1, &yn);
        }
      });

      handle1.join().unwrap();
      handle2.join().unwrap();
    })
    .unwrap();
  }

  fn run_rk4(&mut self) {
    let num_timesteps: usize = (self.total_t / self.dt) as usize;

    debug!("Computing {} timesteps", format_number(num_timesteps));

    let stn_p = self.stn_parameters.clone();
    let gpe_p = self.gpe_parameters.clone();

    let foo = Array1::zeros(10);
    let rho_pre_gpe: &ArrayView1<f64> = &foo.view();
    let rho_pre_stn: &ArrayView1<f64> = &foo.view();

    let dt = self.dt;

    let stn = &mut self.stn_states;
    let gpe = &mut self.gpe_states;

    let mut s_gpe_shared = gpe.s.row(0).to_owned();
    let s_gpe_ref = unsafe { ndarray::ArrayView1::from_shape_ptr(s_gpe_shared.len(), s_gpe_shared.as_ptr()) };
    let mut s_gpe_mut =
      unsafe { ndarray::ArrayViewMut1::from_shape_ptr(s_gpe_shared.len(), s_gpe_shared.as_mut_ptr()) };
    let mut s_stn_shared = stn.s.row(0).to_owned();
    let s_stn_ref = unsafe { ndarray::ArrayView1::from_shape_ptr(s_stn_shared.len(), s_stn_shared.as_ptr()) };
    let mut s_stn_mut =
      unsafe { ndarray::ArrayViewMut1::from_shape_ptr(s_stn_shared.len(), s_stn_shared.as_mut_ptr()) };

    let spin_barrier = Arc::new(SpinBarrier::new(2));
    let barrier1 = spin_barrier.clone();
    let barrier2 = spin_barrier.clone();
    _ = crossbeam::thread::scope(|s| {
      let handle1 = s.spawn(move |_| {
        for it in 0..num_timesteps - 1 {
          let edge_it = it * 2;
          let yp = &stn.row(it);
          barrier1.sym_sync_call(|| s_stn_mut.assign(&yp.s));
          let k1 = yp.dydt(&stn_p, &s_gpe_ref, rho_pre_gpe, &stn.i_ext.row(edge_it));
          let k2 = (yp + &(dt / 2. * &k1)).dydt(&stn_p, &s_gpe_ref, rho_pre_gpe, &stn.i_ext.row(edge_it + 1));
          let k3 = (yp + &(dt / 2. * &k2)).dydt(&stn_p, &s_gpe_ref, rho_pre_gpe, &stn.i_ext.row(edge_it + 1));
          let k4 = (yp + &(dt * &k3)).dydt(&stn_p, &s_gpe_ref, rho_pre_gpe, &stn.i_ext.row(edge_it + 2));
          let yn = yp + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4);
          stn.insert(it + 1, &yn);
        }
      });

      let handle2 = s.spawn(move |_| {
        for it in 0..num_timesteps - 1 {
          let edge_it = it * 2;
          let yp = &gpe.row(it);
          barrier2.sym_sync_call(|| s_gpe_mut.assign(&yp.s));
          let k1 = yp.dydt(&gpe_p, &s_stn_ref, rho_pre_stn, &gpe.i_ext.row(edge_it), &gpe.i_app.row(edge_it));
          let k2 = (yp + &(dt / 2. * &k1)).dydt(
            &gpe_p,
            &s_stn_ref,
            rho_pre_stn,
            &gpe.i_ext.row(edge_it + 1),
            &gpe.i_app.row(edge_it + 1),
          );
          let k3 = (yp + &(dt / 2. * &k2)).dydt(
            &gpe_p,
            &s_stn_ref,
            rho_pre_stn,
            &gpe.i_ext.row(edge_it + 1),
            &gpe.i_app.row(edge_it + 1),
          );
          let k4 = (yp + &(dt * &k3)).dydt(
            &gpe_p,
            &s_stn_ref,
            rho_pre_stn,
            &gpe.i_ext.row(edge_it + 2),
            &gpe.i_app.row(edge_it + 2),
          );
          let yn = yp + dt / 6. * &(k1 + 2. * k2 + 2. * k3 + k4);
          gpe.insert(it + 1, &yn);
        }
      });

      handle1.join().unwrap();
      handle2.join().unwrap();
    })
    .unwrap();

    //  drop(log_tx);
    //  logger_handle.join().unwrap();
  }
}

#[pymethods]
impl Network {
  #[new]
  #[pyo3(signature=(dt=0.01, total_t=2., experiment=None, experiment_version=None, 
                    parameters_file=None, boundry_ic_file=None, use_default=true,
                    save_dir=Some("/tmp/cbgt_last_model".to_owned()), **kwds))]
  fn new_py(
    dt: f64,
    mut total_t: f64,
    experiment: Option<&str>,
    experiment_version: Option<&str>,
    parameters_file: Option<&str>,
    boundry_ic_file: Option<&str>,
    use_default: bool,
    save_dir: Option<String>, // Maybe add datetime to temp save
    kwds: Option<&Bound<'_, PyDict>>,
  ) -> Self {
    total_t *= 1e3;
    assert!(experiment.is_some() || experiment_version.is_none(), "Experiment version requires experiment");

    let save_dir = save_dir.map(|x| x.trim_end_matches("/").to_owned());
    if let Some(save_dir) = &save_dir {
      std::fs::create_dir_all(save_dir).expect("Could not create save folder");
    }

    let experiment = experiment.map(|name| (name, experiment_version));

    let mut stn_kw_config = STNConfig::new();
    let mut gpe_kw_config = GPeConfig::new();

    if let Some(kwds) = kwds {
      for (key, val) in kwds {
        if !(stn_kw_config.update_from_py(&key, &val) || gpe_kw_config.update_from_py(&key, &val)) {
          panic!("Unexpected key word argument {}", key);
        }
      }
    }

    let mut pyf_src = HashMap::from_iter(match Boundary::DEFAULT {
      Some((qname, src)) => vec![(qname.into(), src.into())],
      None => vec![],
    });
    let (stn_kw_params, stn_kw_bcs) = stn_kw_config.into_maps(&mut pyf_src);
    let (gpe_kw_params, gpe_kw_bcs) = gpe_kw_config.into_maps(&mut pyf_src);

    let stn_parameters = BuilderSTNParameters::build(stn_kw_params, parameters_file, experiment, use_default).finish();
    let gpe_parameters = BuilderGPeParameters::build(gpe_kw_params, parameters_file, experiment, use_default).finish();

    log::debug!("{:?}", stn_kw_bcs);
    let mut stn_bcs_builder = BuilderSTNBoundary::build(stn_kw_bcs, boundry_ic_file, experiment, use_default);
    let mut gpe_bcs_builder = BuilderGPeBoundary::build(gpe_kw_bcs, boundry_ic_file, experiment, use_default);

    log::debug!("{:?}", stn_bcs_builder);
    let stn_count = stn_bcs_builder.get_count().expect("stn_count not found");
    let gpe_count = gpe_bcs_builder.get_count().expect("gpe_count not found");

    stn_bcs_builder.extends_pyf_src(&mut pyf_src);
    gpe_bcs_builder.extends_pyf_src(&mut pyf_src);

    let stn_qual_names = stn_bcs_builder.get_callable_qnames();
    let gpe_qual_names = gpe_bcs_builder.get_callable_qnames();
    log::debug!("{:?}", pyf_src);
    let pyf_file = write_temp_pyf_file(pyf_src);

    let num_timesteps: usize = (total_t / dt) as usize;
    let stn_bcs = stn_bcs_builder.finish(stn_count, gpe_count, dt, total_t, 2);
    let gpe_bcs = gpe_bcs_builder.finish(gpe_count, stn_count, dt, total_t, 2);

    if let Some(save_dir) = &save_dir {
      write_parameter_file(&stn_parameters, &gpe_parameters, save_dir);
      write_boundary_file(&stn_bcs, &gpe_bcs, save_dir, &stn_qual_names, &gpe_qual_names);
      std::fs::copy(&pyf_file, format!("{save_dir}/{PYF_FILE_NAME}.py")).unwrap();
    }

    let stn_population = STNPopulation::new(num_timesteps, stn_count, gpe_count, 2).with_bcs(stn_bcs);
    let gpe_population = GPePopulation::new(num_timesteps, stn_count, gpe_count, 2).with_bcs(gpe_bcs);
    let stn_states = STNHistory::from_population(stn_population);
    let gpe_states = GPeHistory::from_population(gpe_population);

    std::fs::remove_file(pyf_file).expect("File was just created, it should exist.");

    Self { dt, total_t, stn_states, stn_parameters, gpe_states, gpe_parameters }
  }

  #[pyo3(name = "run_euler")]
  fn run_euler_py(&mut self) {
    self.run_euler();
  }

  #[pyo3(name = "run_rk4")]
  fn run_rk4_py(&mut self) {
    self.run_rk4();
  }

  #[pyo3(name="to_polars", signature = (dt=None))]
  fn into_map_polars_dataframe_py(&self, py: Python, dt: Option<f64>) -> Py<PyDict> {
    let stn = self.stn_states.into_compressed_polars_df(self.dt, dt, 2);
    let gpe = self.gpe_states.into_compressed_polars_df(self.dt, dt, 2);

    let dict = PyDict::new(py);
    dict.set_item("stn", pyo3_polars::PyDataFrame(stn)).expect("Could not add insert STN Polars DataFrame");
    dict.set_item("gpe", pyo3_polars::PyDataFrame(gpe)).expect("Could not add insert STN Polars DataFrame");

    dict.into()
  }

  #[pyo3(signature = (dir, dt=None))]
  fn save_to_parquet_files(&self, dir: &str, dt: Option<f64>) -> PyResult<()> {
    let dir = std::path::PathBuf::from_str(dir)?;
    std::fs::create_dir_all(&dir)?;

    let mut stn = self.stn_states.into_compressed_polars_df(self.dt, dt, 2);
    let mut gpe = self.gpe_states.into_compressed_polars_df(self.dt, dt, 2);

    let write_parquet = |name: &str, df: &mut polars::prelude::DataFrame| {
      let file_path = dir.join(name);
      let file = std::fs::File::create(&file_path).expect("Could not creare output file");
      let writer = polars::prelude::ParquetWriter::new(&file);
      writer.set_parallel(true).finish(df).expect("Could not write to output file");
      file_path
    };

    write_parquet("stn.parquet", &mut stn);
    write_parquet("gpe.parquet", &mut gpe);

    Ok(())
  }

  #[staticmethod]
  #[pyo3(signature = (log_level="debug"))]
  fn init_logger(log_level: &str) {
    _ = env_logger::Builder::new()
      .filter_level(log_level.parse().expect("Invalid log level"))
      .format(|buf: &mut Formatter, record: &log::Record| writeln!(buf, "[{}] {}", record.level(), record.args()))
      .try_init();
  }
}
