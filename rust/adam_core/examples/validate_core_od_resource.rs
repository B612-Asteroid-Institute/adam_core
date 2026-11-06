//! Validate the public Core OD profile against an explicitly supplied
//! EFCC18 `bias.dat` path. This example never discovers or downloads data.

use std::error::Error;
use std::path::PathBuf;

use adam_core::coords::{ResourceBindings, ResourceSlot, CORE_OD_V1};

fn main() -> Result<(), Box<dyn Error>> {
    let path = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or("usage: validate_core_od_resource /absolute/path/to/bias.dat")?;
    if !path.is_absolute() {
        return Err("bias.dat path must be absolute".into());
    }

    let bindings = ResourceBindings::new().bind(ResourceSlot::Efcc18CatalogBiasTable, path);
    let config = CORE_OD_V1.bind(&bindings)?;
    println!("{}", config.configuration_sha256());
    Ok(())
}
