//! `tetera-server`: serves the Tetera instrument over HTTP, at http://127.0.0.1:8310 by
//! default. `--help` lists its options.

use clap::Parser;
use std::process::ExitCode;

fn main() -> ExitCode {
    tetera_server::run(&tetera_server::Cli::parse())
}
