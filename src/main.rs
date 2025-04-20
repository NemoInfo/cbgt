use cbgt::RubinTerman;

fn main() {
  env_logger::init();

  let mut rt = RubinTerman::new(0.02, 1., Some(("wave", None)), None, None, true, None);
  rt.run();
  println!("Simulation completed!",);
}
