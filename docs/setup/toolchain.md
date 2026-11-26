# Setting up the toolchain

Everything Tetera's build needs, on the Windows host and in WSL2. Each step was run on the owner's development PC on 2026-11-26 (plan §12 had the first draft). `cargo xtask doctor` checks the result.

| When | What |
|---|---|
| Now | WSL2 with Ubuntu 24.04, `.wslconfig`, the system packages, Rust through rustup, cargo-deny, cargo-mutants, the repository in WSL, VS Code over Remote-WSL |
| From P1.04 | The CUDA toolkit for WSL-Ubuntu: P1.04 writes its exact commands |
| From P1.S0 or M5 | usbipd-win, to forward the camera and the stage controller into WSL |
| From M2 | uv, for Python extensions |
| From M4 | Node.js 24 LTS, and Playwright's Chromium |

## On Windows

1. **WSL2 with Ubuntu 24.04:**

   ```powershell
   wsl --install -d Ubuntu-24.04
   ```

   Inside it, systemd must be on: `/etc/wsl.conf` has `[boot]` and `systemd=true`.

2. **The NVIDIA Windows driver.** It gives WSL2 its CUDA, through `/usr/lib/wsl/lib`. Never install a Linux NVIDIA driver inside WSL.

3. **`%UserProfile%\.wslconfig`.** Each key is as [Microsoft's WSL documentation](https://learn.microsoft.com/windows/wsl/wsl-config) gives it. Keep comments on lines of their own: WSL silently ignores a file it can't parse.

   ```ini
   [wsl2]
   # Of the host's 15.7 GB; builds and a 20 MP pipeline need it.
   memory=12GB
   swap=16GB
   processors=16
   # Lets LAN browsers reach the service, when it listens there.
   networkingMode=mirrored

   [general]
   # Never shut the distro down for being idle.
   instanceIdleTimeout=-1

   [experimental]
   autoMemoryReclaim=dropCache
   sparseVhd=true
   ```

   Then restart WSL:

   ```powershell
   wsl --shutdown
   ```

   `sparseVhd` applies to new virtual disks. To make the existing one sparse, run this while WSL is stopped:

   ```powershell
   wsl --manage Ubuntu-24.04 --set-sparse true
   ```

4. **VS Code,** with the WSL extension:

   ```powershell
   code --install-extension ms-vscode-remote.remote-wsl
   ```

   Open the repository with `code --remote wsl+Ubuntu-24.04 /home/<you>/repos/Tetera`, never as a `\\wsl.localhost` folder. The repository's `.vscode/extensions.json` recommends the rest, installed on the WSL side: rust-analyzer, CodeLLDB and Even Better TOML.

5. **usbipd-win,** from P1.S0 or M5, as an administrator:

   ```powershell
   winget install --exact dorssel.usbipd-win
   ```

## In WSL (Ubuntu 24.04)

1. **System packages and git:**

   ```bash
   sudo apt update && sudo apt full-upgrade -y
   sudo apt install -y build-essential pkg-config cmake clang lld mold git curl unzip \
     ca-certificates libudev-dev jq sqlite3 python3-venv gh
   git config --global user.name "<your name>"
   git config --global user.email "<your email>"
   git config --global core.autocrlf input
   ```

2. **Rust, through rustup.** The repository's `rust-toolchain.toml` pins 1.98.1, which rustup installs on the first `cargo` command there:

   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
   . "$HOME/.cargo/env"
   cargo install --locked cargo-deny cargo-mutants
   ```

3. **The repositories,** on WSL's own disk. teta-wot and OScope sit beside Tetera for reference:

   ```bash
   mkdir -p ~/repos && cd ~/repos
   git clone https://github.com/lucaswewa/Tetera.git
   git clone https://github.com/lucaswewa/teta-wot.git
   git clone https://github.com/lucaswewa/OScope.git
   ```

4. **Check it all:**

   ```bash
   cd ~/repos/Tetera
   cargo xtask doctor
   ```

   It lists each tool with `ok`, `warn` or `MISSING`, and ends `Everything needed now is here.` when nothing needed now is missing. Tools for later phases say when they're needed.

## Later

- **The CUDA toolkit (P1.04):** only the `cuda-toolkit-*` package from NVIDIA's WSL-Ubuntu repository; the `cuda` and `cuda-drivers` packages would install a Linux driver.
- **uv (M2):** `curl -LsSf https://astral.sh/uv/install.sh | sh`.
- **Node.js 24 LTS (M4):** through nvm, then `npx playwright install --with-deps chromium`.
