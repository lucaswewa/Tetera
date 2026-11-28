# Reaching Tetera from Windows

Tetera's server runs in WSL2 ([ADR-0018](../../docs/adr/0018-the-service-in-wsl2-systemd-start-up-networking-and-devices.md)). P5.07 hardens what follows into a bootstrap; for development it is done by hand.

## A browser on the same PC

Open <http://localhost:8310>, or <http://localhost:8310/docs> for the interactive API docs.

- **With mirrored networking** (`networkingMode=mirrored` in `%UserProfile%\.wslconfig`, as [the toolchain's setup](../../docs/setup/toolchain.md) has it), WSL shares Windows's network interfaces: `localhost` is the same on both sides.
- **With WSL's default NAT networking,** WSL forwards `localhost` ports to Windows (`localhostForwarding`, on by default), so the same URL works.

The server listens on `127.0.0.1` (Q11): browsers on other PCs can't reach it. Opening it to the lab's network is M5's, configured (`[server] address`), with a firewall rule.

## Starting WSL at logon

WSL2 starts only when something on Windows starts it, and the service starts with the distribution. Two settings keep it there:

1. **In `%UserProfile%\.wslconfig`,** `instanceIdleTimeout=-1` under `[general]`, so the distribution isn't shut down for being idle.
2. **A task that starts the distribution at logon.** In PowerShell:

   ```powershell
   $action = New-ScheduledTaskAction -Execute "wsl.exe" -Argument "-d Ubuntu-24.04 --exec /bin/true"
   $trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
   Register-ScheduledTask -TaskName "Tetera WSL" -Action $action -Trigger $trigger -Description "Starts WSL, and with it Tetera's service"
   ```

   `Unregister-ScheduledTask -TaskName "Tetera WSL"` removes it. Starting the instrument at boot without a login (Q12) is P5.07's.
