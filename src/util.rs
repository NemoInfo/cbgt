use ndarray::{Array, ArrayView, Dimension};

use std::sync::atomic::{AtomicUsize, Ordering};

pub fn x_inf<D: Dimension>(v: &ArrayView<f64, D>, tht_x: f64, sig_x: f64) -> Array<f64, D> {
  1. / (1. + ((tht_x - v) / sig_x).exp())
}

pub fn tau_x<D: Dimension>(
  v: &ArrayView<f64, D>,
  tau_x_0: f64,
  tau_x_1: f64,
  tht_x_t: f64,
  sig_x_t: f64,
) -> Array<f64, D> {
  tau_x_0 + tau_x_1 / (1. + ((tht_x_t - v) / sig_x_t).exp())
}

pub struct SpinBarrier {
  count: AtomicUsize,
  generation: AtomicUsize,
  total: usize,
}

impl SpinBarrier {
  pub fn new(total: usize) -> Self {
    Self { count: AtomicUsize::new(total), generation: AtomicUsize::new(0), total }
  }

  pub fn wait(&self) {
    // Record the current generation.
    let gen = self.generation.load(Ordering::Relaxed);
    // Decrement the counter.
    if self.count.fetch_sub(1, Ordering::AcqRel) == 1 {
      // Last thread to arrive.
      self.count.store(self.total, Ordering::Release);
      // Advance to the next generation.
      self.generation.fetch_add(1, Ordering::Release);
    } else {
      // Wait until the generation changes.
      while self.generation.load(Ordering::Acquire) == gen {
        std::hint::spin_loop();
      }
    }
  }
}
