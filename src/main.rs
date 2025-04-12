use cbgt::RubinTerman;

// use ndarray_npy::write_npy;
// use pyo3::Python;

fn main() {
  env_logger::init();

  let _ = RubinTerman::new(10, 10, 0.02, 2., true, None, None)._run();
  println!("Simulation completed!",);

  // @TODO: Fix serialization find something nice to serialize a HashMap<HashMap<Array>> to
  //        maybe ?pickle?
  // let file_name = "test.npy";
  // write_npy(&file_name, &res).expect(&format!("Failed to write results to {file_name}"));
  // println!("Wrote results to {file_name}!");
}
