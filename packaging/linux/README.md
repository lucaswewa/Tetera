# Tetera as a systemd service in WSL

[`tetera.service`](tetera.service) runs Tetera's server under systemd in Ubuntu 24.04 in WSL2 ([ADR-0018](../../docs/adr/0018-the-service-in-wsl2-systemd-start-up-networking-and-devices.md)). It is the development install; P5.07 packages it as a `.deb`.

- **It runs** `/usr/local/bin/tetera-server` as a user of its own (`DynamicUser`), with its data in `/var/lib/tetera` and its configuration in `/etc/tetera/`.
- **It restarts on failure:** killed, crashed, or stopped on an error, it is started again 2 s later. A clean stop shuts it down gracefully, and it stays down.
- **Its logs go to journald** (`journalctl -u tetera`), which the server detects.
- **It needs systemd in WSL:** `/etc/wsl.conf` has `[boot] systemd=true`, as [the toolchain's setup](../../docs/setup/toolchain.md) makes it.

## Installing it for development

The owner runs these, as they need `sudo`. From the repository's root in WSL:

```bash
cargo build --release -p tetera-server
sudo install -m 755 target/release/tetera-server /usr/local/bin/tetera-server
sudo install -d /etc/tetera
sudo install -m 644 configs/tetera.toml configs/objectives.toml /etc/tetera/
sudo install -m 644 packaging/linux/tetera.service /etc/systemd/system/tetera.service
sudo systemctl daemon-reload
sudo systemctl enable --now tetera
```

Then:

```bash
systemctl status tetera
curl http://127.0.0.1:8310/system/version
journalctl -u tetera -f
```

**To update it,** build and `install` the binary again, then `sudo systemctl restart tetera`.

**To see it restart,** kill it: `sudo systemctl kill -s KILL tetera`. `systemctl status tetera` shows it running again a few seconds later, with a new PID.

**To remove it:** `sudo systemctl disable --now tetera`, then delete the files installed above. `/var/lib/tetera` holds its data.

## As a user unit, without `sudo`

For trying the unit without installing anything, the same unit runs as a user service, from the repository's build:

```bash
mkdir -p ~/.config/systemd/user
sed -e "s|/usr/local/bin/tetera-server|$PWD/target/release/tetera-server|" \
    -e "s|/etc/tetera/tetera.toml|$PWD/configs/tetera.toml|" \
    -e "s|/var/lib/tetera|%S/tetera|g" \
    -e "/^DynamicUser=/d" \
    -e "s|multi-user.target|default.target|" \
    packaging/linux/tetera.service > ~/.config/systemd/user/tetera.service
systemctl --user daemon-reload
systemctl --user start tetera
```

`%S` is `~/.local/state` for a user unit. `systemctl --user kill -s KILL tetera` shows the restart; `systemctl --user stop tetera` stops it.
