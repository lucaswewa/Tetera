# Architecture decision records

One record per significant decision, numbered across the project, from the [template](template.md). A phase's ADRs are marked *Accepted* in its pull request: approving the pull request accepts them. A later ADR that changes one marks it *Superseded*.

The [plan's §10](../implementation-plan.md#10-planned-adrs) lists the ADRs still to come, and the phase that writes each.

| ADR | Title | Status | Phase |
|---|---|---|---|
| [0001](0001-recording-decisions-and-the-delivery-process.md) | Recording decisions, and the delivery process | Accepted | P1.01 |
| [0002](0002-clean-room-and-dependency-licences.md) | Clean room and dependency licences | Accepted | P1.01 |
| [0003](0003-the-runtime-platform-ubuntu-24-04-in-wsl2.md) | The runtime platform: Ubuntu 24.04 in WSL2 | Accepted | P1.01 |
| [0004](0004-teta-wot-v0-2-0-extended-upstream-first.md) | teta-wot v0.2.0, extended upstream first | Accepted | P1.01 |
| [0005](0005-layered-crates-and-the-local-gates.md) | Layered crates, and the local gates | Accepted | P1.01 |
| [0006](0006-units-coordinates-and-the-clock.md) | Units, coordinates and the clock | Accepted | P1.02 |
| [0007](0007-logging.md) | Logging | Accepted | P1.02 |
| [0008](0008-the-frame-model-the-buffer-pool-and-the-frame-bus.md) | The frame model, the buffer pool and the frame bus | Accepted | P1.03 |
| [0009](0009-compute-backends-the-cpu-reference-and-cuda.md) | Compute backends: the CPU reference and CUDA, and what WSL2 offers | Accepted | P1.04a |
| [0010](0010-building-and-loading-cuda-kernels-under-wsl2.md) | Building and loading CUDA kernels under WSL2 | Accepted | P1.04b |
| [0011](0011-fused-kernels.md) | Fused kernels | Accepted | P1.06b |
| [0012](0012-the-pipeline-model.md) | The pipeline model | Accepted | P1.08a |
| [0013](0013-the-hardware-abstraction-layer.md) | The hardware abstraction layer | Accepted | P1.09a |
| [0014](0014-the-simulated-clock.md) | The simulated clock | Accepted | P1.10a |
| [0015](0015-simulated-samples-on-the-gpu.md) | Simulated samples on the GPU | Accepted | P1.11a |
| [0016](0016-the-virtual-cameras-model.md) | The virtual camera's model | Accepted | P1.12a |
| [0017](0017-control-plane-and-media-plane.md) | Control plane and media plane | Accepted | P1.13a |
| [0018](0018-the-service-in-wsl2-systemd-start-up-networking-and-devices.md) | The service in WSL2: systemd, start-up, networking and devices | Accepted | P1.13a |
| [0019](0019-the-preview-protocol.md) | The preview protocol | Accepted | P1.14b |
| [0020](0020-the-job-service.md) | The job service | Accepted | P1.15a |
| [0021](0021-datasets-on-disk.md) | Datasets on disk | Accepted | P1.15b |
| [0022](0022-scan-planning-and-the-scan-engine.md) | Scan planning and the scan engine | Accepted | P1.17a |
