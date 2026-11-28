# ADR-0018: The service in WSL2: systemd, start-up, networking and devices

- Status: Accepted
- Date: 2026-11-28
- Phase: P1.13a

## Context

- **Tetera runs in Ubuntu 24.04 in WSL2** (ADR-0003), with systemd enabled in `/etc/wsl.conf`.
- **The instrument must keep running:** after a crash, after it is killed, and, from M5, after Windows boots without a login (Q12).
- **WSL2 starts only when Windows starts it,** and shuts an idle distribution down unless told not to (plan R7).
- **Browsers run on Windows.** Same PC only in M1–M4; the lab's network from M5, configured (Q11). The port is 8310, beside OScope's 8210 (Q48).
- **Devices:** the GPU reaches WSL through `/dev/dxg` and `/usr/lib/wsl/lib`; USB devices, the camera and the stage, only through `usbipd-win` (M5).
- **Data must stay on ext4:** `/mnt/c` goes through a network file system that is far too slow (plan §6.1).
- **The owner runs anything that needs `sudo`;** the agent can't.

## Decision

1. **A systemd unit, `packaging/linux/tetera.service`:**
   - it runs `/usr/local/bin/tetera-server` as a user of its own (`DynamicUser`), with `--config /etc/tetera/tetera.toml` and `--data-root /var/lib/tetera`, which systemd makes (`StateDirectory`);
   - **`Restart=on-failure`, 2 s later:** a kill, a crash, or an exit on an error restarts it; a clean stop doesn't;
   - a clean stop is SIGTERM, which the server takes as its shutdown signal: unfinished invocations are cancelled, and the Things stopped, within 5 s; systemd kills it after 15;
   - its logs go to journald, which `tetera-log` detects under systemd.
2. **Development installs the unit by hand,** with the steps in `packaging/linux/README.md`, which the owner runs. The same unit runs as a user unit without `sudo`, as the agent tests it. P5.07 packages it as a `.deb`.
3. **Start-up:** `instanceIdleTimeout=-1` in `.wslconfig` keeps the distribution running, and a Windows task at logon starts it (`packaging/windows/README.md`). Starting at boot without a login is P5.07's.
4. **Networking:** the server listens on `127.0.0.1:8310`. Mirrored networking (`networkingMode=mirrored`) shares Windows's `localhost`; with NAT, WSL's `localhostForwarding` does the same. Listening on the lab's network is a configured address, and a firewall rule, from M5.
5. **Data roots:** `./.tetera/` in development, `/var/lib/tetera/` for the service, both on ext4; the scratch root inside the data root unless configured.
6. **Devices:** the GPU as WSL gives it, found at start-up, with the CPU as the fallback; USB devices through `usbipd-win` in M5.

## Consequences

- **Gains:**
  - the service comes back by itself after a failure, and stops cleanly when asked;
  - a browser on the same PC reaches it at `http://localhost:8310`, whichever networking mode WSL uses;
  - development and the service read the same configuration, with the data root overridden.
- **Costs:**
  - an install by hand until P5.07, and only after a login;
  - `DynamicUser` keeps the service's files private to it: the owner reads them with `sudo`.
- **To watch:**
  - whether WSL keeps the service's distribution alive in every case with `instanceIdleTimeout=-1`;
  - whether CUDA works for a dynamic user as it does for the owner (Checkpoint C).

## Alternatives considered

- **A Windows service that starts WSL:** what P5.07 may need for boot without a login, but more than development wants.
- **Running in the user's session (`systemctl --user`):** no `sudo`, but it ends with the user's session unless lingering is on, and the instrument isn't a user's program.
- **`Restart=always`:** a clean stop would restart it too.
- **Listening on every address in development:** reachable from the lab's network before M5 says how, against Q11.
