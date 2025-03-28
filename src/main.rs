use cbgt::RubinTerman;

// use ndarray_npy::write_npy;
// use pyo3::Python;

fn main() {
  let _ = RubinTerman {
    dt: 0.01,
    total_t: 2.,
    ..Default::default()
  }
  ._run();
  println!("Simulation completed!");

  // [[FIX SERIALISATION]]
  // let file_name = "test.npy";
  // write_npy(&file_name, &res).expect(&format!("Failed to write results to {file_name}"));
  // println!("Wrote results to {file_name}!");
}
