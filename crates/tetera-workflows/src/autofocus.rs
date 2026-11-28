//! Autofocus on the instrument (P1.16b): P1.16a's search through the stage's z, each position
//! measured on the first frame of the live stream exposed after the stage reported itself
//! still there, so a moving stage never blurs a measurement. The behaviour spec is P1.16a's.

use crate::focus::{Focus, FocusProbe, SearchOptions, search};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tetera_acquire::FrameBus;
use tetera_core::{Error, Result, StagePos, Timestamp, Um};
use tetera_hal::camera::{Camera, FrameSink};
use tetera_hal::objective::Objective;
use tetera_hal::stage::{Axis, Stage};
use tetera_imaging::Mosaic;
use tetera_imaging::focus::FocusMeasure;

/// How long a z move may take to end.
const MOVE: Duration = Duration::from_secs(10);

/// How long a frame exposed after a move may take to come.
const FRAME: Duration = Duration::from_secs(5);

/// The light's wavelength the depth of field is worked out at, µm.
pub const WAVELENGTH_UM: f64 = 0.55;

/// The depth of field, µm, of an objective of `magnification` and numerical aperture `na`
/// on a sensor of `pixel_um` pixels: `λ / NA² + e / (M · NA)` (P1.16a's spec).
pub fn depth_of_field(magnification: f64, na: f64, pixel_um: f64) -> f64 {
    WAVELENGTH_UM / (na * na) + pixel_um / (magnification * na)
}

/// What autofocus is asked for.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(default, deny_unknown_fields)]
pub struct AutofocusRequest {
    /// The focus measure: normalised variance when not given.
    pub measure: FocusMeasure,
    /// How the search goes, in depths of field.
    pub options: SearchOptions,
}

/// What autofocus did.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct AutofocusResult {
    /// The search: the focus, or why there is none, and its curve.
    pub focus: Focus,
    /// The measure it used.
    pub measure: FocusMeasure,
    /// The objective it focused.
    pub objective: String,
    /// How long it took, s.
    pub seconds: f64,
}

/// What autofocus works with.
pub struct FocusRig<'a> {
    /// The camera, whose live stream is measured.
    pub camera: &'a dyn Camera,
    /// The stage, whose z is searched.
    pub stage: &'a dyn Stage,
    /// The frame bus the live stream goes into.
    pub bus: &'a Arc<FrameBus>,
    /// Where the live view's frames go, to start it if it isn't running.
    pub live: Option<Arc<dyn FrameSink>>,
    /// The objective fitted.
    pub objective: Option<Objective>,
    /// The sensor's pixel pitch, µm.
    pub sensor_pixel_um: f64,
}

/// Focuses the instrument as `request` asks: the stage left at the focus, or, with none
/// found, where it started. The live stream is started for it if it wasn't running, and
/// stopped after. `step(done, message)` is called as it goes: an error from it stops it.
pub fn autofocus(
    rig: &FocusRig<'_>,
    request: &AutofocusRequest,
    step: &mut dyn FnMut(f64, &str) -> Result<()>,
) -> Result<AutofocusResult> {
    request.options.check()?;
    let objective = rig.objective.clone().ok_or_else(|| {
        Error::invalid_argument("autofocus needs an objective fitted, to know its depth of field")
    })?;
    let d = depth_of_field(
        objective.magnification,
        objective.numerical_aperture,
        rig.sensor_pixel_um,
    );
    let travel = rig.stage.capabilities().axis(Axis::Z)?.travel;
    let began = Instant::now();
    let started = !rig.camera.is_streaming();
    if started {
        let sink: Arc<dyn FrameSink> = match &rig.live {
            Some(live) => live.clone(),
            None => rig.bus.clone(),
        };
        rig.camera.start(sink)?;
    }
    let searched = StageProbe::new(rig.stage, rig.bus, request.measure).and_then(|mut probe| {
        search(
            &mut probe,
            d,
            (travel.min, travel.max),
            &request.options,
            step,
        )
    });
    let stopped = if started { rig.camera.stop() } else { Ok(()) };
    let focus = searched?;
    stopped?;
    Ok(AutofocusResult {
        focus,
        measure: request.measure,
        objective: objective.name,
        seconds: began.elapsed().as_secs_f64(),
    })
}

/// The instrument as the search's probe: z moves of the stage, x and y kept, and the measure
/// of the first frame exposed after each move ended.
pub(crate) struct StageProbe<'a> {
    stage: &'a dyn Stage,
    bus: &'a Arc<FrameBus>,
    measure: FocusMeasure,
    x: Um,
    y: Um,
    /// When the stage last reported itself still, by its clock.
    still: Timestamp,
    /// For each measure: when the stage was still, and when its frame's exposure started.
    pub(crate) taken: Vec<(Timestamp, Timestamp)>,
}

impl<'a> StageProbe<'a> {
    pub(crate) fn new(
        stage: &'a dyn Stage,
        bus: &'a Arc<FrameBus>,
        measure: FocusMeasure,
    ) -> Result<Self> {
        let reading = stage.reading()?;
        Ok(Self {
            stage,
            bus,
            measure,
            x: reading.position.x,
            y: reading.position.y,
            still: reading.at,
            taken: Vec::new(),
        })
    }
}

impl FocusProbe for StageProbe<'_> {
    fn z(&mut self) -> Result<f64> {
        Ok(self.stage.reading()?.position.z.um())
    }

    fn move_to(&mut self, z: f64) -> Result<()> {
        self.stage
            .move_to(StagePos::new(self.x, self.y, Um(z)), None)?;
        self.still = self.stage.wait(MOVE)?.at;
        Ok(())
    }

    fn measure(&mut self) -> Result<f64> {
        let frame = self.bus.first_after(self.still, FRAME)?;
        self.taken.push((self.still, frame.meta().exposure_start));
        self.measure.of_mosaic(&Mosaic::from_frame(&frame)?)
    }
}

#[cfg(test)]
mod tests;
