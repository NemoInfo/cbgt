use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use env_logger::fmt::Formatter;
use indicatif::ProgressBar;
use indicatif::ProgressStyle;
use log::debug;
use ndarray::Array2;
use pyo3::{prelude::*, types::PyDict};
use std::io::Write;

use crate::ctx::BuilderCTXBoundary;
use crate::ctx::CTXConfig;
use crate::ctx::CTXHistory;
use crate::gpe::*;
use crate::gpi::*;
use crate::parameters::*;
use crate::stn::*;
use crate::str::BuilderSTRBoundary;
use crate::str::STRConfig;
use crate::str::STRHistory;
use crate::types::{Boundary, NeuronData};
use crate::util::wrap_idx;
use crate::util::{format_number, write_boundary_file, write_parameter_file, write_temp_pyf_file, SpinBarrier};
use crate::PYF_FILE_NAME;
use crate::{gpe::GPeHistory, GPeParameters};
use crate::{stn::STNHistory, STNParameters};

#[pyclass]
pub struct Network {
  #[pyo3(get)]
  pub dt: f64,
  #[pyo3(get)]
  pub total_time: f64,
  pub str: STRHistory,
  pub str_p: STRParameters,
  pub stn: STNHistory,
  pub stn_p: STNParameters,
  pub gpe: GPeHistory,
  pub gpe_p: GPeParameters,
  pub gpi: GPiHistory,
  pub gpi_p: GPiParameters,
  pub ctx: CTXHistory,
  pub ctx_p: CTXParameters,
  pub batch_duration: f64,
}

struct RunOutput {
  stn: polars::frame::DataFrame,
  str: polars::frame::DataFrame,
  gpe: polars::frame::DataFrame,
  gpi: polars::frame::DataFrame,
  ctx: polars::frame::DataFrame,
}

impl Network {
  fn run_rk4(&mut self, data: &Vec<&str>, odt: Option<f64>) -> RunOutput {
    let total_timesteps = (self.total_time / odt.unwrap_or(1.)) as usize;
    let batches = (self.total_time / self.batch_duration).ceil() as usize - 1;
    _ = self.run_rk4_batch(0., self.batch_duration);
    let mut stn = self.stn.into_compressed_polars_df(self.dt, odt, 2, data, 0.);
    let mut str = self.str.into_compressed_polars_df(self.dt, odt, 2, data, 0.);
    let mut gpe = self.gpe.into_compressed_polars_df(self.dt, odt, 2, data, 0.);
    let mut gpi = self.gpi.into_compressed_polars_df(self.dt, odt, 2, data, 0.);
    let mut ctx = self.ctx.into_compressed_polars_df(self.dt, odt, 2, data, 0.);
    for batch in 1..batches + 1 {
      let start_time = self.batch_duration * batch as f64;
      self.roll();
      _ = self.run_rk4_batch(start_time, (start_time + self.batch_duration).min(self.total_time));
      stn = stn.vstack(&self.stn.into_compressed_polars_df(self.dt, odt, 2, data, start_time)).unwrap();
      str = str.vstack(&self.str.into_compressed_polars_df(self.dt, odt, 2, data, start_time)).unwrap();
      gpe = gpe.vstack(&self.gpe.into_compressed_polars_df(self.dt, odt, 2, data, start_time)).unwrap();
      gpi = gpi.vstack(&self.gpi.into_compressed_polars_df(self.dt, odt, 2, data, start_time)).unwrap();
      ctx = ctx.vstack(&self.ctx.into_compressed_polars_df(self.dt, odt, 2, data, start_time)).unwrap();
    }

    stn.rechunk_mut();
    str.rechunk_mut();
    gpe.rechunk_mut();
    gpi.rechunk_mut();
    ctx.rechunk_mut();

    stn = stn.slice(0, total_timesteps);
    str = str.slice(0, total_timesteps);
    gpe = gpe.slice(0, total_timesteps);
    gpi = gpi.slice(0, total_timesteps);
    ctx = ctx.slice(0, total_timesteps);

    RunOutput { stn, str, gpe, gpi, ctx }
  }

  fn roll(&mut self) {
    self.stn.roll();
    self.str.roll();
    self.gpe.roll();
    self.gpi.roll();
  }

  fn run_rk4_batch(&mut self, start_time: f64, end_time: f64) -> (Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>) {
    let Self { dt, total_time, str, str_p, stn, stn_p, gpe, gpe_p, gpi, gpi_p, ctx, ctx_p, batch_duration } = self;
    let dt = *dt;

    let duration = end_time - start_time;
    let num_timesteps: usize = (duration / dt) as usize;
    let batch_idx = (start_time / *batch_duration) as usize + 1;
    let batches = (*total_time / *batch_duration).ceil() as usize;

    debug!("Computing {} timesteps", format_number(num_timesteps));

    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
      str.fill_i_ext(py, start_time, end_time, dt / 2.);
      stn.fill_i_ext(py, start_time, end_time, dt / 2.);
      gpe.fill_i_ext(py, start_time, end_time, dt / 2.);
      gpi.fill_i_ext(py, start_time, end_time, dt / 2.);
      ctx.fill_s(py, ctx_p, start_time, end_time, dt / 2.);
    });

    let mut dd_stn = DiracDeltaState::new(stn.v.raw_dim());
    let d_stn_ref = unsafe { ndarray::ArrayView2::from_shape_ptr(dd_stn.d.raw_dim(), dd_stn.d.as_ptr()) };
    let mut dd_gpe = DiracDeltaState::new(gpe.v.raw_dim());
    let d_gpe_ref = unsafe { ndarray::ArrayView2::from_shape_ptr(dd_gpe.d.raw_dim(), dd_gpe.d.as_ptr()) };
    let mut dd_gpi = DiracDeltaState::new(gpi.v.raw_dim());
    let d_gpi_ref = unsafe { ndarray::ArrayView2::from_shape_ptr(dd_gpi.d.raw_dim(), dd_gpi.d.as_ptr()) };
    let mut d_gpi_mut = unsafe { ndarray::ArrayViewMut2::from_shape_ptr(dd_gpi.d.raw_dim(), dd_gpi.d.as_mut_ptr()) };
    let mut dd_str = DiracDeltaState::new(str.v.raw_dim());
    let d_str_ref = unsafe { ndarray::ArrayView2::from_shape_ptr(dd_str.d.raw_dim(), dd_str.d.as_ptr()) };

    let stn_s = unsafe { ndarray::ArrayView2::from_shape_ptr(stn.s.raw_dim(), stn.s.as_ptr()) };
    let str_s = unsafe { ndarray::ArrayView2::from_shape_ptr(str.s.raw_dim(), str.s.as_ptr()) };
    let gpe_s = unsafe { ndarray::ArrayView2::from_shape_ptr(gpe.s.raw_dim(), gpe.s.as_ptr()) };
    let _gpi_s = unsafe { ndarray::ArrayView2::from_shape_ptr(gpi.s.raw_dim(), gpi.s.as_ptr()) };
    let ctx_s = unsafe { ndarray::ArrayView2::from_shape_ptr(ctx.s.raw_dim(), ctx.s.as_ptr()) };

    let spin_barrier = Arc::new(SpinBarrier::new(4));
    let stn_barrier = spin_barrier.clone();
    let gpe_barrier = spin_barrier.clone();
    let gpi_barrier = spin_barrier.clone();
    let str_barrier = spin_barrier.clone();
    let (dd_stn, dd_gpe, dd_gpi, dd_str) = crossbeam::thread::scope(|s| {
      let stn_thread = s.spawn(move |_| {
        let pb = ProgressBar::new(num_timesteps as u64 - 1);
        pb.set_style(
          ProgressStyle::default_bar()
            .template(
              &(format!("BATCH {batch_idx}/{batches} ")
                + "[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})"),
            )
            .unwrap()
            .progress_chars("=> "),
        );

        for it in 0..num_timesteps - 1 {
          if it % 100 == 0 {
            pb.set_position(it as u64);
          }
          let ctx_id = if batch_idx != 1 {
            it.wrapping_sub((1. / dt) as usize) as isize // NOTE This can panick for very small batch durations, it can loop around twice
          } else {
            it.saturating_sub((1. / dt) as usize) as isize
          };
          let ctx_s = ctx_s.row(wrap_idx(ctx_id, num_timesteps) * 2);
          let edge_it = it * 2;
          let yp = &stn.row(it);
          stn_barrier.sym_sync_call(|| dd_stn.update(&yp.v, dt, it));
          d_gpi_mut.row_mut(it).assign(&stn.ca_g_s.row(0));

          let d_stn = d_stn_ref.row(it);
          let d_gpe = d_gpe_ref.row(it);
          let gpe_s = gpe_s.row(it);

          let (k1, is) = yp.dydt(&stn_p, &d_stn, &d_gpe, &gpe_s, &ctx_s, &stn.i_ext.row(edge_it));
          let (k2, _) =
            (yp + &(dt / 2. * &k1)).dydt(&stn_p, &d_stn, &d_gpe, &gpe_s, &ctx_s, &stn.i_ext.row(edge_it + 1));
          let (k3, _) =
            (yp + &(dt / 2. * &k2)).dydt(&stn_p, &d_stn, &d_gpe, &gpe_s, &ctx_s, &stn.i_ext.row(edge_it + 1));
          let (k4, _) = (yp + &(dt * &k3)).dydt(&stn_p, &d_stn, &d_gpe, &gpe_s, &ctx_s, &stn.i_ext.row(edge_it + 2));
          let yn = yp + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4);
          stn.insert(it + 1, &yn, is);
        }
        pb.finish_and_clear();
        dd_stn
      });

      let str_thread = s.spawn(move |_| {
        for it in 0..num_timesteps - 1 {
          let edge_it = it * 2;
          let ctx_id = if batch_idx != 1 {
            it.wrapping_sub((10.5 / dt) as usize) as isize // NOTE This can panick for very small batch durations, it can loop around twice
          } else {
            it.saturating_sub((10.5 / dt) as usize) as isize
          };

          let ctx_s = ctx_s.row(wrap_idx(ctx_id, num_timesteps) * 2);
          let yp = &str.row(it);
          str_barrier.sym_sync_call(|| dd_str.update(&yp.v, dt, it));

          let k1 = yp.dydt(&str_p, &ctx_s, &str.i_ext.row(edge_it));
          let k2 = (yp + &(dt / 2. * &k1)).dydt(&str_p, &ctx_s, &str.i_ext.row(edge_it + 1));
          let k3 = (yp + &(dt / 2. * &k2)).dydt(&str_p, &ctx_s, &str.i_ext.row(edge_it + 1));
          let k4 = (yp + &(dt * &k3)).dydt(&str_p, &ctx_s, &str.i_ext.row(edge_it + 2));
          let yn = yp + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4);
          str.insert(it + 1, &yn);
        }
        dd_str
      });

      let gpe_thread = s.spawn(move |_| {
        for it in 0..num_timesteps - 1 {
          let edge_it = it * 2;
          let yp = &gpe.row(it);
          gpe_barrier.sym_sync_call(|| {
            dd_gpe.update(&yp.v, dt, it);
          });

          let d_stn = d_stn_ref.row(it);
          let d_gpe = d_gpe_ref.row(it);
          let d_str = d_str_ref.row(it);
          let stn_s = stn_s.row(it);
          let str_s = str_s.row(it);

          let (k1, is) =
            yp.dydt(&gpe_p, &d_gpe, &d_stn, &d_str, &stn_s, &str_s, &gpe.i_ext.row(edge_it), &gpe.i_app.row(edge_it));
          let (k2, _) = (yp + &(dt / 2. * &k1)).dydt(
            &gpe_p,
            &d_gpe,
            &d_stn,
            &d_str,
            &stn_s,
            &str_s,
            &gpe.i_ext.row(edge_it + 1),
            &gpe.i_app.row(edge_it + 1),
          );
          let (k3, _) = (yp + &(dt / 2. * &k2)).dydt(
            &gpe_p,
            &d_gpe,
            &d_stn,
            &d_str,
            &stn_s,
            &str_s,
            &gpe.i_ext.row(edge_it + 1),
            &gpe.i_app.row(edge_it + 1),
          );
          let (k4, _) = (yp + &(dt * &k3)).dydt(
            &gpe_p,
            &d_gpe,
            &d_stn,
            &d_str,
            &stn_s,
            &str_s,
            &gpe.i_ext.row(edge_it + 2),
            &gpe.i_app.row(edge_it + 2),
          );
          let yn = yp + dt / 6. * &(k1 + 2. * k2 + 2. * k3 + k4);
          gpe.insert(it + 1, &yn, is);
        }
        dd_gpe
      });

      let gpi_thread = s.spawn(move |_| {
        for it in 0..num_timesteps - 1 {
          let edge_it = it * 2;
          let yp = &gpi.row(it);
          gpi_barrier.sym_sync_call(|| dd_gpi.update(&yp.v, dt, it));

          let d_gpi = d_gpi_ref.row(it);
          let d_stn = d_stn_ref.row(it);
          let d_str = d_str_ref.row(it);
          let stn_s = stn_s.row(it);
          let str_s = str_s.row(it);

          let k1 =
            yp.dydt(&gpi_p, &d_gpi, &d_stn, &d_str, &stn_s, &str_s, &gpi.i_ext.row(edge_it), &gpi.i_app.row(edge_it));
          let k2 = (yp + &(dt / 2. * &k1)).dydt(
            &gpi_p,
            &d_gpi,
            &d_stn,
            &d_str,
            &stn_s,
            &str_s,
            &gpi.i_ext.row(edge_it + 1),
            &gpi.i_app.row(edge_it + 1),
          );
          let k3 = (yp + &(dt / 2. * &k2)).dydt(
            &gpi_p,
            &d_gpi,
            &d_stn,
            &d_str,
            &stn_s,
            &str_s,
            &gpi.i_ext.row(edge_it + 1),
            &gpi.i_app.row(edge_it + 1),
          );
          let k4 = (yp + &(dt * &k3)).dydt(
            &gpi_p,
            &d_gpi,
            &d_stn,
            &d_str,
            &stn_s,
            &str_s,
            &gpi.i_ext.row(edge_it + 2),
            &gpi.i_app.row(edge_it + 2),
          );
          let yn = yp + dt / 6. * &(k1 + 2. * k2 + 2. * k3 + k4);
          gpi.insert(it + 1, &yn);
        }
        dd_gpi
      });

      let dd_stn = stn_thread.join().unwrap();
      let dd_gpe = gpe_thread.join().unwrap();
      let dd_gpi = gpi_thread.join().unwrap();
      let dd_str = str_thread.join().unwrap();

      (dd_stn, dd_gpe, dd_gpi, dd_str)
    })
    .unwrap();

    (dd_stn.d, dd_gpe.d, dd_gpi.d, dd_str.d)
  }
}

#[pymethods]
impl Network {
  #[new]
  #[pyo3(signature=(dt=0.01, total_time=2., experiment=None, experiment_version=None, 
                    parameters_file=None, boundry_ic_file=None, use_default=true,
                    save_dir=Some("/tmp/cbgt_last_model".to_owned()), batch_duration=4f64, **kwds))]
  fn new_py(
    dt: f64,
    mut total_time: f64,
    experiment: Option<&str>,
    experiment_version: Option<&str>,
    parameters_file: Option<&str>,
    boundry_ic_file: Option<&str>,
    use_default: bool,
    save_dir: Option<String>, // Maybe add datetime to temp save
    mut batch_duration: f64,  // Batch duration in s
    kwds: Option<&Bound<'_, PyDict>>,
  ) -> Self {
    total_time *= 1e3;
    batch_duration *= 1e3;
    assert!(experiment.is_some() || experiment_version.is_none(), "Experiment version requires experiment");

    let save_dir = save_dir.map(|x| x.trim_end_matches("/").to_owned());
    if let Some(save_dir) = &save_dir {
      std::fs::create_dir_all(save_dir).expect("Could not create save folder");
    }

    let experiment = experiment.map(|name| (name, experiment_version));

    let mut stn_kw_config = STNConfig::new();
    let mut gpe_kw_config = GPeConfig::new();
    let mut gpi_kw_config = GPiConfig::new();
    let mut str_kw_config = STRConfig::new();
    let mut ctx_kw_config = CTXConfig::new();

    if let Some(kwds) = kwds {
      for (key, val) in kwds {
        if !(stn_kw_config.update_from_py(&key, &val)
          || gpe_kw_config.update_from_py(&key, &val)
          || gpi_kw_config.update_from_py(&key, &val)
          || str_kw_config.update_from_py(&key, &val)
          || ctx_kw_config.update_from_py(&key, &val))
        {
          panic!("Unexpected key word argument {}", key);
        }
      }
    }

    let mut pyf_src = HashMap::from_iter(match Boundary::DEFAULT {
      Some((qname, src)) => vec![(qname.into(), src.into())],
      None => vec![],
    });

    let (str_kw_params, str_kw_bcs) = str_kw_config.into_maps(&mut pyf_src);
    let (stn_kw_params, stn_kw_bcs) = stn_kw_config.into_maps(&mut pyf_src);
    let (gpe_kw_params, gpe_kw_bcs) = gpe_kw_config.into_maps(&mut pyf_src);
    let (gpi_kw_params, gpi_kw_bcs) = gpi_kw_config.into_maps(&mut pyf_src);
    let (ctx_kw_params, ctx_kw_bcs) = ctx_kw_config.into_maps(&mut pyf_src);

    let str_p = BuilderSTRParameters::build(str_kw_params, parameters_file, experiment, use_default).finish();
    let stn_p = BuilderSTNParameters::build(stn_kw_params, parameters_file, experiment, use_default).finish();
    let gpe_p = BuilderGPeParameters::build(gpe_kw_params, parameters_file, experiment, use_default).finish();
    let gpi_p = BuilderGPiParameters::build(gpi_kw_params, parameters_file, experiment, use_default).finish();
    let ctx_p = BuilderCTXParameters::build(ctx_kw_params, parameters_file, experiment, use_default).finish();

    log::debug!("{:?}", ctx_kw_bcs);
    let mut str_bcs_builder = BuilderSTRBoundary::build(str_kw_bcs, boundry_ic_file, experiment, use_default);
    let mut stn_bcs_builder = BuilderSTNBoundary::build(stn_kw_bcs, boundry_ic_file, experiment, use_default);
    let mut gpe_bcs_builder = BuilderGPeBoundary::build(gpe_kw_bcs, boundry_ic_file, experiment, use_default);
    let mut gpi_bcs_builder = BuilderGPiBoundary::build(gpi_kw_bcs, boundry_ic_file, experiment, use_default);
    let mut ctx_bcs_builder = BuilderCTXBoundary::build(ctx_kw_bcs, boundry_ic_file, experiment, use_default);

    log::debug!("{:?}", ctx_bcs_builder);
    let str_count = str_bcs_builder.get_count().expect("str_count not found");
    let stn_count = stn_bcs_builder.get_count().expect("stn_count not found");
    let gpe_count = gpe_bcs_builder.get_count().expect("gpe_count not found");
    let gpi_count = gpi_bcs_builder.get_count().expect("gpi_count not found");
    let ctx_count = ctx_bcs_builder.get_count().expect("ctx_count not found");

    str_bcs_builder.extends_pyf_src(&mut pyf_src);
    stn_bcs_builder.extends_pyf_src(&mut pyf_src);
    gpe_bcs_builder.extends_pyf_src(&mut pyf_src);
    gpi_bcs_builder.extends_pyf_src(&mut pyf_src);
    ctx_bcs_builder.extends_pyf_src(&mut pyf_src);

    let str_qual_names = str_bcs_builder.get_callable_qnames();
    let stn_qual_names = stn_bcs_builder.get_callable_qnames();
    let gpe_qual_names = gpe_bcs_builder.get_callable_qnames();
    let gpi_qual_names = gpi_bcs_builder.get_callable_qnames();
    let ctx_qual_names = ctx_bcs_builder.get_callable_qnames();
    log::debug!("{:?}", pyf_src);
    let pyf_file = write_temp_pyf_file(pyf_src);

    let num_timesteps: usize = (batch_duration / dt) as usize;
    println!("{num_timesteps}");
    let str_bcs = str_bcs_builder.finish(str_count, ctx_count);
    let stn_bcs = stn_bcs_builder.finish(stn_count, gpe_count, ctx_count);
    let gpe_bcs = gpe_bcs_builder.finish(gpe_count, stn_count, str_count);
    let gpi_bcs = gpi_bcs_builder.finish(gpi_count, stn_count, str_count);
    let ctx_bcs = ctx_bcs_builder.finish(ctx_count);

    if let Some(save_dir) = &save_dir {
      // @TODO save gpi and str as well
      write_parameter_file(&stn_p, &gpe_p, save_dir); // TODO add other nuclei
      write_boundary_file(
        &stn_bcs,
        &gpe_bcs,
        &gpi_bcs,
        &str_bcs,
        &ctx_bcs,
        save_dir,
        &stn_qual_names,
        &gpe_qual_names,
        &gpi_qual_names,
        &str_qual_names,
        &ctx_qual_names,
      );
      std::fs::copy(&pyf_file, format!("{save_dir}/{PYF_FILE_NAME}.py")).unwrap();
    }

    let str = STRHistory::new(num_timesteps, str_count, ctx_count, 2, str_bcs);
    let stn = STNHistory::new(num_timesteps, stn_count, gpe_count, ctx_count, 2, stn_bcs);
    let gpe = GPeHistory::new(num_timesteps, gpe_count, stn_count, str_count, 2, gpe_bcs);
    let gpi = GPiHistory::new(num_timesteps, gpi_count, stn_count, str_count, 2, gpi_bcs);
    // let ctx = ctx_bcs.to_syn_ctx(&ctx_p, dt / 2.);
    let ctx = CTXHistory::new(num_timesteps, ctx_count, 2, ctx_bcs);

    std::fs::remove_file(pyf_file).expect("File was just created, it should exist.");

    Self { dt, total_time, stn, stn_p, gpe, gpe_p, gpi, gpi_p, str, str_p, ctx, ctx_p, batch_duration }
  }

  #[pyo3(name = "run_rk4", signature = (data=vec![], odt=None, save_dir=Some("test_out/".into())))]
  fn run_rk4_py<'py>(
    &mut self,
    py: Python,
    data: Vec<String>,
    odt: Option<f64>,
    save_dir: Option<String>,
  ) -> Py<PyDict> {
    let data = data.iter().map(String::as_str).collect();
    let RunOutput { mut stn, mut str, mut gpe, mut gpi, mut ctx } = self.run_rk4(&data, odt);

    if let Some(save_dir) = save_dir {
      let save_dir = PathBuf::from(save_dir);

      let write_parquet = |name: &str, df: &mut polars::prelude::DataFrame| {
        let file_path = save_dir.join(name);
        let file = std::fs::File::create(&file_path).expect("Could not creare output file");
        let writer = polars::prelude::ParquetWriter::new(&file);
        writer.set_parallel(true).finish(df).expect("Could not write to output file");
        file_path
      };

      write_parquet("str.parquet", &mut str);
      write_parquet("stn.parquet", &mut stn);
      write_parquet("gpe.parquet", &mut gpe);
      write_parquet("gpi.parquet", &mut gpi);
      write_parquet("ctx.parquet", &mut ctx);
    }

    let dict = PyDict::new(py);
    dict.set_item("stn", pyo3_polars::PyDataFrame(stn)).expect("Could not add insert STN Polars DataFrame");
    dict.set_item("str", pyo3_polars::PyDataFrame(str)).expect("Could not add insert STR Polars DataFrame");
    dict.set_item("gpe", pyo3_polars::PyDataFrame(gpe)).expect("Could not add insert GPe Polars DataFrame");
    dict.set_item("gpi", pyo3_polars::PyDataFrame(gpi)).expect("Could not add insert GPi Polars DataFrame");
    dict.set_item("ctx", pyo3_polars::PyDataFrame(ctx)).expect("Could not add insert CTX Polars DataFrame");

    dict.into()
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

#[cfg(test)]
mod size_test {}
