//! Resolve every adam-core data file and report its source.
//!
//! ```sh
//! cargo run -p adam_core_rs_kernel_data --example resolve_all
//! ```

use adam_core_rs_kernel_data::{Resolver, KERNEL_SPECS};

fn main() {
    let resolver = Resolver::from_env();
    let mut failures = 0usize;
    for spec in KERNEL_SPECS {
        match resolver.resolve(spec.id) {
            Ok(path) => {
                let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                println!("{:14} {:>12} bytes  {}", spec.id, size, path.display());
            }
            Err(error) => {
                failures += 1;
                eprintln!("{:14} ERROR: {error}", spec.id);
            }
        }
    }
    if failures > 0 {
        std::process::exit(1);
    }
}
