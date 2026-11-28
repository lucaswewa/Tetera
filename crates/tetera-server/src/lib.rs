//! Tetera's server, L6 (plan §6.2): the binary that serves the instrument over HTTP. It reads
//! `configs/tetera.toml` ([`config`]), takes the command line's overrides, sets up logging,
//! makes the data roots, assembles the virtual instrument and serves its Things (through
//! `tetera-api`) until it is asked to stop, then stops gracefully.

pub mod config;

use clap::Parser;
use config::Config;
use std::future::Future;
use std::net::{IpAddr, SocketAddr};
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use tetera_api::{Instrument, Roots};
use tetera_core::{Error, Result};
use tokio::net::TcpListener;

/// The command line.
#[derive(Debug, Clone, Default, Parser)]
#[command(
    name = "tetera-server",
    version,
    about = "Serves the Tetera instrument over HTTP."
)]
pub struct Cli {
    /// The configuration file; without one, `configs/tetera.toml` if it is there, else the
    /// defaults.
    #[arg(short, long)]
    pub config: Option<PathBuf>,
    /// The address to listen on, over the configuration's.
    #[arg(long)]
    pub address: Option<IpAddr>,
    /// The port to listen on, over the configuration's.
    #[arg(long)]
    pub port: Option<u16>,
    /// The data root, over the configuration's: `/var/lib/tetera` for the service.
    #[arg(long)]
    pub data_root: Option<PathBuf>,
    /// Checks the configuration, prints it with the overrides, and exits.
    #[arg(long)]
    pub check: bool,
}

impl Cli {
    /// The configuration it asks for: the file's, with its overrides.
    pub fn config(&self) -> Result<Config> {
        let (mut config, _) = Config::find(self.config.as_deref())?;
        if let Some(address) = self.address {
            config.server.address = address;
        }
        if let Some(port) = self.port {
            config.server.port = port;
        }
        if let Some(root) = &self.data_root {
            config.data.root = root.clone();
        }
        Ok(config)
    }
}

/// Runs the server as `cli` asks, until a shutdown signal: its exit code.
pub fn run(cli: &Cli) -> ExitCode {
    let config = match cli.config() {
        Ok(config) => config,
        Err(e) => {
            eprintln!("tetera-server: {e}");
            return ExitCode::from(2);
        }
    };
    if cli.check {
        println!("{config:#?}");
        return ExitCode::SUCCESS;
    }
    let _logs = match tetera_log::init(&config.log) {
        Ok(logs) => logs,
        Err(e) => {
            eprintln!("tetera-server: logging: {e}");
            return ExitCode::from(2);
        }
    };
    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(e) => {
            eprintln!("tetera-server: the runtime won't start: {e}");
            return ExitCode::FAILURE;
        }
    };
    let address = SocketAddr::new(config.server.address, config.server.port);
    let served = runtime.block_on(async move {
        let listener = TcpListener::bind(address)
            .await
            .map_err(|e| Error::invalid_argument(format!("{address} can't be listened on: {e}")))?;
        serve(config, listener, tetera_api::shutdown_signal()).await
    });
    match served {
        Ok(()) => {
            tracing::info!("stopped");
            ExitCode::SUCCESS
        }
        Err(e) => {
            tracing::error!(error = %e, "the server stopped on an error");
            eprintln!("tetera-server: {e}");
            ExitCode::FAILURE
        }
    }
}

/// Serves the instrument `config` describes on `listener` until `shutdown` completes: the
/// data roots made, the instrument assembled, its Things served, then stopped gracefully.
pub async fn serve(
    config: Config,
    listener: TcpListener,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> Result<()> {
    let roots = Roots::make(config.data.root.clone(), config.data.scratch())?;
    let (simulator, objectives) = (config.simulator.clone(), config.objectives.clone());
    let instrument = tokio::task::spawn_blocking(move || Instrument::new(&simulator, &objectives))
        .await
        .map_err(|e| Error::internal(format!("the instrument's assembly failed: {e}")))??;
    let compute = instrument.compute();
    let local = listener
        .local_addr()
        .map_err(|e| Error::internal(format!("the listener has no address: {e}")))?;
    tracing::info!(
        url = %format!("http://{local}"),
        backend = %compute.backend,
        sensor = ?compute.sensor,
        data = %roots.data.display(),
        "serving"
    );
    let service = tetera_api::build_with(Arc::new(instrument), roots, &config.jobs)?;
    service.serve(listener, shutdown).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use tetera_api::SensorSize;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpStream;

    /// `GET path` on `address`: the status line and the body.
    async fn get(address: SocketAddr, path: &str) -> (String, String) {
        let mut stream = TcpStream::connect(address).await.unwrap();
        let request =
            format!("GET {path} HTTP/1.1\r\nHost: {address}\r\nConnection: close\r\n\r\n");
        stream.write_all(request.as_bytes()).await.unwrap();
        let mut answer = String::new();
        stream.read_to_string(&mut answer).await.unwrap();
        let status = answer.lines().next().unwrap_or_default().to_string();
        let body = answer
            .split("\r\n\r\n")
            .nth(1)
            .unwrap_or_default()
            .to_string();
        (status, body)
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn it_serves_on_a_port_the_system_picks_and_stops_when_asked() {
        let root = std::env::temp_dir().join(format!("tetera-server-test-{}", std::process::id()));
        let mut config = Config::default();
        config.data.root = root.clone();
        config.simulator.gpu = false;
        config.simulator.sensor = SensorSize::Tiny;
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
        let server = tokio::spawn(serve(config, listener, async move {
            let _ = stopped.await;
        }));
        let (status, body) = get(address, "/system/version").await;
        assert!(status.contains("200"), "{status}");
        assert!(body.contains(env!("CARGO_PKG_VERSION")), "{body}");
        let (status, _) = get(address, "/simulation/").await;
        assert!(status.contains("200"), "{status}");
        assert!(
            root.join("data").exists() || root.exists(),
            "the data root made"
        );
        assert!(root.join("scratch").is_dir(), "the scratch root made");
        stop.send(()).unwrap();
        let finished = tokio::time::timeout(std::time::Duration::from_secs(10), server)
            .await
            .expect("it stops within its grace period");
        finished.unwrap().unwrap();
        assert!(
            TcpStream::connect(address).await.is_err(),
            "the port let go"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn the_command_line_overrides_the_configuration() {
        let cli = Cli::try_parse_from([
            "tetera-server",
            "--config",
            concat!(env!("CARGO_MANIFEST_DIR"), "/../../configs/tetera.toml"),
            "--port",
            "9100",
            "--address",
            "0.0.0.0",
            "--data-root",
            "/var/lib/tetera",
        ])
        .unwrap();
        let config = cli.config().unwrap();
        assert_eq!(config.server.port, 9100);
        assert_eq!(config.server.address.to_string(), "0.0.0.0");
        assert_eq!(config.data.root, PathBuf::from("/var/lib/tetera"));
        assert_eq!(
            config.data.scratch(),
            PathBuf::from("/var/lib/tetera/scratch")
        );
        assert!(Cli::try_parse_from(["tetera-server", "--port", "port"]).is_err());
        let missing = Cli {
            config: Some("/nowhere.toml".into()),
            ..Cli::default()
        };
        assert!(missing.config().is_err(), "a configuration that won't open");
    }
}
