//! Prints Tetera's name and version: the smallest example, for trying the gates and the
//! debugger.
//!
//! Run it with `cargo run -p tetera-core --example hello`. With `-- --smoke` it does the
//! same and says so, as every example's smoke mode must finish quickly and exit 0.

fn main() {
    let smoke = std::env::args().any(|arg| arg == "--smoke");
    let greeting = tetera_core::name_and_version();
    println!("{greeting}");
    if smoke {
        println!("smoke: ok");
    }
}
