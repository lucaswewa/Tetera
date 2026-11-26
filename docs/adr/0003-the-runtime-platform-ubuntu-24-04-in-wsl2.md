# ADR-0003: The runtime platform: Ubuntu 24.04 in WSL2

- Status: Accepted
- Date: 2026-11-26
- Phase: P1.01

## Context

- **The compute module is a Windows PC,** and the spec runs the instrument service inside WSL2's Ubuntu 24.04, with CUDA from the Windows driver.
- **OScope built and tested for Windows and Linux.** Every gate ran twice, and each platform's differences cost phases rounds of fixes.
- **Building from a Windows drive inside WSL** (`/mnt/d`) is many times slower than from WSL's own ext4 disk.
- **WSL2 changes some things:** USB devices arrive only through usbipd-win; WSL stops when idle unless told not to; `systemctl reboot` restarts the VM, not Windows; the wall clock can drift after the host sleeps.

## Decision

1. **Tetera runs and is built on Linux x86_64 only:** Ubuntu 24.04 in WSL2, under systemd. No native Windows build (plan D2, Q8).
2. **Windows hosts** WSL, the browser, USB forwarding with usbipd-win (M5), and, only if USB forwarding falls short, a small camera bridge (plan R1).
3. **The repository lives in WSL's ext4 file system,** `~/repos/Tetera`, edited from VS Code over Remote-WSL (plan D3).
4. **`.wslconfig`** gives WSL enough memory and keeps it running: `docs/setup/toolchain.md` lists the settings.
5. **Platform-specific code** is Linux code. What touches Windows (power actions, start-up, USB forwarding) goes through WSL's interop or Windows tasks, in the phases that need it (P1.13, P1.20, M5).

## Consequences

- **Gains:** one gate run per change; one set of system calls; the GPU, the disk and the service behave as they will on the instrument.
- **Costs:** the camera must reach WSL through USB forwarding, which may not keep up (P1.S0 measures it); power and start-up need Windows-side help.
- **To watch:** USB forwarding's rate; WSL's memory under builds; the virtual disk's growth.

## Alternatives considered

- **A native Windows service,** as OScope offered: the spec puts the service in WSL2, and two platforms double every gate.
- **Both platforms:** the same cost, kept only for a fallback the owner placed in a Windows camera bridge instead.
- **Containers inside WSL:** another layer between the service, the GPU and USB, for nothing the spec asks.
