use pyo3::prelude::*;

mod parameters;
use parameters::*;

mod stn;

mod str;

mod gpe;

mod gpi;

mod util;

mod types;

mod ctx;

mod network;
use network::Network;

pub const TMP_PYF_FILE_PATH: &'static str = ".";
pub const TMP_PYF_FILE_NAME: &'static str = "temp_functions";
pub const PYF_FILE_NAME: &'static str = "functions";

#[pymodule]
fn cbgt(m: &Bound<'_, PyModule>) -> PyResult<()> {
  m.add_class::<Network>()?;
  Ok(())
}
