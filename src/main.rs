use cbgt::RubinTerman;

use ndarray_npy::write_npy;

fn main() {
  let res = RubinTerman {
    dt: 0.01,
    total_t: 2.,
    ..Default::default()
  }
  ._run();
  println!("Simulation completed!");

  let file_name = "test.npy";
  write_npy(&file_name, &res).expect(&format!("Failed to write results to {file_name}"));
  println!("Wrote results to {file_name}!");
}
