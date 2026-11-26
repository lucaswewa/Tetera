# Scientific Instrument Software: Build Specification

I want to build a professional scientific instrument software for lab applications. In this specific build, I want to use microscope as the example instrument.

## Requirements

The followings are the requirements:

* The software is branded as `Tetera`
* Hardware summary:
  + Camera: MIchrome 20 USB3.1. The microscope has a camera `MIChrome 20`(https://www.tucsen.com/michrome-20-product/), color, +esolution 20MP, 5472(H) x 3648(V), pixel size: 2.4 μm x 2.4 μm, 14.5 fps @ 5472x3648, bit depth: 16/8bit
  + Stage: xyz motorized stage, using grblHAL over serial port. xy can be used for scanning, z can be used for focusing or z-scan.
  + The compute module: this is a Windows PC with WSL2 Ubuntu-24.04 setup. The instrument software runs in the WSL2 Ubuntu-24.04 as service. The compute module also has an Nvidia GPU, where CUDA is used for image processing.
* Software architecture and basic requirements:
  + Use Rust backend consisting of layered architecture.
  + The Web frontend handles user interaction. Vue3 + Vite should be used as frontend framework.
  + Streaming strategy: Live view (Preview) - In Rust, downsamplle and compress the 12-bit raw frame into a lightweight WebP/JPEG or H.+64 and send the lightweight compressed preview frames to the UI as 30-60 fps.
  + Raw data (for analysis/stitching/saving, etc): Keep the prestine 12-bit raw data strictly in Rust memor or flush it directly to a +igh-speed NVMe scratch disk.
  + Image processing
    - Processing pipeline (Rust): a pipeline of nodes for image processing in Rust. CUDA GPU is used when available, otherwise, fall -ack to CPU.
    - The processing nodes (or fused CUDA kernel): stock algorithms are provided with the instrument software. Extensions framework is -uilt to support 3rd party nodes as extensibility.
    - Stock processing nodes include: Bayer raw data splitting to RGB; denoise; dark field subtraction; flat field; deconvolution PSF; -olor-correction matrix CCM; raw image to JPEG/MJPEG.
    - Extension processing nodes include, not limited to: Color space conversion: XYZ->LAB; object recognition; 2D deep-zoom; stitching; depth-from-focus; 3D volume; etc.
  + workflow
    - basic workflows include: capture; scanning (xy snake scan, z-scan); auto-focus; calibration.
    - Extension workflows include: Raster scan; Tiled z-stack; position scan; Histo scanning;
  + Use `teta-wot`(https://github.com/lucaswewa/teta-wot) as the HTTP interface
  + Use `OScope`(https://github.com/lucaswewa/OScope) as reference project
  + Backend
    - GPU first, then fallback to CPU if GPU is not avaiable
    - HAL hardware abstract layer
    - Real hardware: Camera, Stage, Objectives
    - Virtual hardware: VirtualCamera, VirtualStage
    - Simulation samples: H&E tissue;
    - Simulation illumination
    - Support Extensions mechanism: Rust + Python
    - `teta-wot v0.2.0` as the HTTP interface
  + Frontend
    - Vue3+Typescript+Vite
    - Use VTK.js to render 3D.
    - Use the reka-ui framework
    - Use pinia for state management
    - use material-symbols
    - use themes and accent color in `OScope`
    - UI major features:
      * A navRail consists of tabs: `Control and Live View`, `Scan workflow`, `Gallery`, `Workspace`, `Settings`, `Logging`, `About`, `Power`, `Status Indicator`.
      * `Control and Live View` consists of: camera control; stage control; live view
      * `Scan workflow` consists of: xy scan; z-scan; etc
      * `Gallery`: this is raw data (single capture, multiple capture, scanning) stored and displayed as card.
      * `Workspace`: users can process the data, make measurement, perform analysis, generate reports.
      * `Settings`: settings for System, display, camera, stage, calibration, wizard, simulated sample settings, simulated samples load/unload.
      * `Logging`: info, warn, error, critical, etc.
      * `About`: about the software
      * `Power`: restart theh service, reboot the host machine, shutdown.
      * `Status Indicator`: status and health condition
* Lessons learned from `OScope`
  + implement the software in smaller milestone with multiple phases:
    - M1: focus on the Rust backend, supporting core functionalities
    - M2: focus on the Rust backend, supporting Extensions
    - M3: focus on the Rust backend, supporting 3rd nodes and workflow using Extensions
    - M4: focus on the Vue3 frontend, supporting virtual hardware and simulation samples
    - M5: focus on integration with real hardware and performance fine tune.
  + Use GPU to generated simulation samples and virtual camera image capture and live view.
  + Every phase must include ADRs, implementation notes, manual test, examples
  + Don't run CI actions on GitHub as gate, but instead, run them on local machine. Divide the gate into smaller ones, such that they run quickly for each phase. Run the entire CI gates at the end of the milestone.
  + List the instructions of install toolchains on local machine and WSL2.

