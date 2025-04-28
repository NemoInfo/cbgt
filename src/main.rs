use cbgt::RubinTerman;

fn main() {
  env_logger::init();

  let mut rt = RubinTerman::new(0.01, 2., Some(("wave", None)), None, None, true, None);
  rt.run();
  let _ = rt.into_map_polars_dataframe(None);
  println!("Simulation completed!",);
}
