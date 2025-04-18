use cbgt::RubinTerman;

fn main() {
  env_logger::init();

  let mut rt = RubinTerman::new(10, 10, 0.02, 1., true, Some(("wave", None)), None);
  rt.run();
  rt.into_map_polars_dataframe(None);
  println!("Simulation completed!",);
}
