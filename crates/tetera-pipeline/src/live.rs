//! Live runs (P1.08d): a plan run on a thread of its own, one frame at a time. A frame offered
//! while one runs waits; a frame offered while one waits takes its place. So no more than one
//! frame ever waits, and the next to run is always the latest.

use crate::node::Context;
use crate::plan::Plan;
use crate::port::Value;
use std::collections::BTreeMap;
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::thread::JoinHandle;
use tetera_compute::Compute;
use tetera_core::{Error, Result};

/// What a live run has done.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LiveStats {
    /// Frames offered.
    pub offered: u64,
    /// Frames run.
    pub ran: u64,
    /// Frames that waited, and gave their place to a later one.
    pub replaced: u64,
    /// A frame still waiting when the run was stopped.
    pub dropped: u64,
}

/// A run's outputs, or why it failed.
pub type Outputs = Result<BTreeMap<String, Value>>;

#[derive(Debug, Default)]
struct State {
    waiting: Option<Value>,
    stop: bool,
    stats: LiveStats,
}

type Shared = Arc<(Mutex<State>, Condvar)>;

fn lock(shared: &Shared) -> std::sync::MutexGuard<'_, State> {
    shared.0.lock().unwrap_or_else(PoisonError::into_inner)
}

/// A plan running live.
#[derive(Debug)]
pub struct Live {
    input: String,
    ty: crate::port::PortType,
    shared: Shared,
    thread: Option<JoinHandle<Plan>>,
}

impl Live {
    /// `plan` run live on `compute`, each frame offered as input `input`; every other input
    /// must be held. Each run's outputs, or its error, go to `deliver`, on the live thread.
    pub fn start(
        plan: Plan,
        compute: Arc<Compute>,
        input: &str,
        mut deliver: impl FnMut(Outputs) + Send + 'static,
    ) -> Result<Self> {
        let Some(k) = plan.inputs.iter().position(|(n, _)| n == input) else {
            return Err(Error::invalid_argument(format!(
                "recipe `{}` has no input `{input}` to run live",
                plan.name
            )));
        };
        let ty = plan.inputs[k].1;
        if let Some((name, _)) = plan
            .inputs
            .iter()
            .zip(&plan.held)
            .find(|((n, _), held)| n != input && held.is_none())
            .map(|(i, _)| i)
        {
            return Err(Error::invalid_argument(format!(
                "recipe `{}` takes input `{name}`, which a live run must hold: it gives `{input}` alone",
                plan.name
            )));
        }
        if plan.held[k].is_some() {
            return Err(Error::invalid_argument(format!(
                "recipe `{}` holds input `{input}`; let it go to run it live",
                plan.name
            )));
        }
        let shared: Shared = Arc::default();
        let thread = {
            let shared = shared.clone();
            let input = input.to_string();
            let mut plan = plan;
            std::thread::Builder::new()
                .name(format!("live {}", plan.name))
                .spawn(move || {
                    loop {
                        let frame = {
                            let mut state = lock(&shared);
                            while state.waiting.is_none() && !state.stop {
                                state =
                                    shared.1.wait(state).unwrap_or_else(PoisonError::into_inner);
                            }
                            if state.stop {
                                break;
                            }
                            state.waiting.take().expect("a frame waits")
                        };
                        let inputs = BTreeMap::from([(input.clone(), frame)]);
                        let outputs = plan.run(&Context { compute: &compute }, inputs);
                        lock(&shared).stats.ran += 1;
                        deliver(outputs);
                    }
                    plan
                })
                .map_err(|e| Error::internal(format!("couldn't start a live run: {e}")))?
        };
        Ok(Self {
            input: input.to_string(),
            ty,
            shared,
            thread: Some(thread),
        })
    }

    /// `frame` to run next, in place of any frame waiting: whether one was waiting.
    pub fn offer(&self, frame: Value) -> Result<bool> {
        if frame.port_type() != self.ty {
            return Err(Error::invalid_argument(format!(
                "a live run takes {} as `{}`, not {}",
                self.ty,
                self.input,
                frame.port_type()
            )));
        }
        let mut state = lock(&self.shared);
        state.stats.offered += 1;
        let replaced = state.waiting.replace(frame).is_some();
        if replaced {
            state.stats.replaced += 1;
        }
        self.shared.1.notify_one();
        Ok(replaced)
    }

    /// What it has done so far.
    pub fn stats(&self) -> LiveStats {
        lock(&self.shared).stats
    }

    /// Whether every frame offered has run, or given its place: none runs or waits.
    pub fn is_idle(&self) -> bool {
        let stats = lock(&self.shared).stats;
        stats.ran + stats.replaced == stats.offered
    }

    /// Stops it: the frame running finishes, a frame waiting is dropped. It gives the plan back,
    /// and what it did.
    pub fn stop(mut self) -> Result<(Plan, LiveStats)> {
        let plan = self.halt()?;
        Ok((plan, self.stats()))
    }

    fn halt(&mut self) -> Result<Plan> {
        {
            let mut state = lock(&self.shared);
            state.stop = true;
            if state.waiting.take().is_some() {
                state.stats.dropped += 1;
            }
            self.shared.1.notify_all();
        }
        let thread = self.thread.take().expect("running until stopped");
        thread
            .join()
            .map_err(|_| Error::internal("the live run's thread panicked"))
    }
}

impl Drop for Live {
    fn drop(&mut self) {
        if self.thread.is_some() {
            let _ = self.halt();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::{Descriptor, NoParameters, Node, Step, parameters};
    use crate::port::{PortSpec, PortType};
    use crate::recipe::Recipe;
    use crate::registry::Registry;
    use std::sync::mpsc;
    use std::time::{Duration, Instant};

    /// `test.gate` runs a frame when the test sends it a token, and gives the frame plus its
    /// held offset; a negative frame is an error.
    struct Gate(Descriptor, Mutex<Option<mpsc::Receiver<()>>>);

    struct GateStep(mpsc::Receiver<()>);

    impl Node for Gate {
        fn descriptor(&self) -> &Descriptor {
            &self.0
        }

        fn prepare(&self, params: &serde_json::Value) -> Result<Box<dyn Step>> {
            let _: NoParameters = parameters(params)?;
            let tokens = self.1.lock().unwrap().take().expect("prepared once");
            Ok(Box::new(GateStep(tokens)))
        }
    }

    impl Step for GateStep {
        fn run(&mut self, _: &Context, inputs: &[&Value]) -> Result<Vec<Value>> {
            self.0
                .recv_timeout(Duration::from_secs(10))
                .map_err(|_| Error::internal("no token for the gate"))?;
            let [Value::Scalar(frame), Value::Scalar(offset)] = inputs else {
                return Err(Error::internal("the gate takes two scalars"));
            };
            if *frame < 0.0 {
                return Err(Error::invalid_argument("a negative frame"));
            }
            Ok(vec![Value::Scalar(frame + offset)])
        }
    }

    /// A plan of one gate, its offset held, and the gate's tokens.
    fn gated(offset: Option<f64>) -> (Plan, mpsc::Sender<()>) {
        let (tx, rx) = mpsc::channel();
        let mut registry = Registry::new();
        let scalar = |name| PortSpec::new(name, PortType::Scalar, "a number");
        let d = Descriptor::new("test.gate", 1, "gated")
            .input(scalar("frame"))
            .input(scalar("offset"))
            .output(scalar("sum"))
            .stateful();
        registry
            .register(Arc::new(Gate(d, Mutex::new(Some(rx)))))
            .unwrap();
        let recipe = Recipe::from_json(
            r#"{"name": "gated", "inputs": {"frame": {"kind": "scalar"}, "offset": {"kind": "scalar"}},
            "nodes": [{"id": "gate", "node": "test.gate", "inputs": {"frame": "frame", "offset": "offset"}}],
            "outputs": {"sum": "gate.sum"}}"#,
        )
        .unwrap();
        let mut plan = Plan::new(&recipe, &registry).unwrap();
        if let Some(offset) = offset {
            plan.hold("offset", Value::Scalar(offset)).unwrap();
        }
        (plan, tx)
    }

    fn compute() -> Arc<Compute> {
        Arc::new(Compute::cpu_only().unwrap())
    }

    /// Waits for `done`, for up to 10 s.
    fn wait_for(live: &Live, done: impl Fn(&Live) -> bool) {
        let start = Instant::now();
        while !done(live) {
            assert!(
                start.elapsed() < Duration::from_secs(10),
                "{:?}",
                live.stats()
            );
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    fn taken(live: &Live) -> bool {
        lock(&live.shared).waiting.is_none()
    }

    fn sums(rx: &mpsc::Receiver<Outputs>) -> Vec<Result<f64>> {
        rx.try_iter()
            .map(|out| {
                out.map(|o| match o["sum"] {
                    Value::Scalar(s) => s,
                    _ => unreachable!(),
                })
            })
            .collect()
    }

    #[test]
    fn frames_offered_while_one_runs_give_way_to_the_latest() {
        let (plan, tokens) = gated(Some(1_000.0));
        let (tx, rx) = mpsc::channel();
        let live = Live::start(plan, compute(), "frame", move |out| tx.send(out).unwrap()).unwrap();
        assert!(!live.offer(Value::Scalar(1.0)).unwrap());
        wait_for(&live, taken);
        // While frame 1 runs, 19 more come: one waits at a time, each taking its place.
        assert!(
            !live.offer(Value::Scalar(2.0)).unwrap(),
            "nothing waiting yet"
        );
        for frame in 3..=20 {
            assert!(
                live.offer(Value::Scalar(f64::from(frame))).unwrap(),
                "frame {frame}"
            );
            assert!(lock(&live.shared).waiting.is_some());
        }
        tokens.send(()).unwrap();
        tokens.send(()).unwrap();
        wait_for(&live, Live::is_idle);
        let stats = live.stats();
        assert_eq!(
            stats,
            LiveStats {
                offered: 20,
                ran: 2,
                replaced: 18,
                dropped: 0
            }
        );
        let (plan, last) = live.stop().unwrap();
        assert_eq!(last, stats);
        let got: Vec<f64> = sums(&rx).into_iter().map(Result::unwrap).collect();
        assert_eq!(got, [1_001.0, 1_020.0], "the first, then the latest");
        assert_eq!(plan.name(), "gated", "the plan given back");
    }

    #[test]
    fn frames_offered_after_each_run_all_run_and_errors_are_delivered() {
        let (plan, tokens) = gated(Some(0.0));
        let (tx, rx) = mpsc::channel();
        let live = Live::start(plan, compute(), "frame", move |out| tx.send(out).unwrap()).unwrap();
        for frame in [1.0, 2.0, -1.0, 4.0] {
            assert!(
                !live.offer(Value::Scalar(frame)).unwrap(),
                "nothing waiting"
            );
            tokens.send(()).unwrap();
            wait_for(&live, Live::is_idle);
        }
        assert_eq!(
            live.stats(),
            LiveStats {
                offered: 4,
                ran: 4,
                replaced: 0,
                dropped: 0
            }
        );
        let got = sums(&rx);
        assert_eq!(got.len(), 4);
        assert_eq!(
            (
                got[0].as_ref().unwrap(),
                got[1].as_ref().unwrap(),
                got[3].as_ref().unwrap()
            ),
            (&1.0, &2.0, &4.0)
        );
        assert_eq!(got[2].as_ref().unwrap_err().to_string(), "a negative frame");
    }

    #[test]
    fn a_frame_waiting_at_the_stop_is_dropped() {
        let (plan, tokens) = gated(Some(0.0));
        let (tx, rx) = mpsc::channel();
        let live = Live::start(plan, compute(), "frame", move |out| tx.send(out).unwrap()).unwrap();
        live.offer(Value::Scalar(1.0)).unwrap();
        wait_for(&live, taken);
        live.offer(Value::Scalar(2.0)).unwrap();
        // Frame 1 finishes once the stop is asked for, so frame 2 never starts.
        let shared = live.shared.clone();
        let release = std::thread::spawn(move || {
            while !lock(&shared).stop {
                std::thread::sleep(Duration::from_millis(1));
            }
            tokens.send(()).unwrap();
        });
        let (_, stats) = live.stop().unwrap();
        release.join().unwrap();
        assert_eq!(
            stats,
            LiveStats {
                offered: 2,
                ran: 1,
                replaced: 0,
                dropped: 1
            }
        );
        assert_eq!(sums(&rx).len(), 1);
    }

    #[test]
    fn dropping_a_live_run_stops_its_thread() {
        let (plan, _tokens) = gated(Some(0.0));
        let (tx, rx) = mpsc::channel::<Outputs>();
        let live = Live::start(plan, compute(), "frame", move |out| tx.send(out).unwrap()).unwrap();
        drop(live);
        // The thread has ended, and its callback with it, so the channel has closed.
        assert_eq!(
            rx.recv_timeout(Duration::from_secs(5)).unwrap_err(),
            mpsc::RecvTimeoutError::Disconnected
        );
    }

    #[test]
    fn a_frame_of_the_wrong_type_is_refused_as_it_is_offered() {
        let (plan, _tokens) = gated(Some(0.0));
        let live = Live::start(plan, compute(), "frame", |_| {}).unwrap();
        assert_eq!(
            live.offer(Value::Table(Default::default()))
                .unwrap_err()
                .to_string(),
            "a live run takes a scalar as `frame`, not a table"
        );
        assert_eq!(live.stats().offered, 0);
        assert!(live.is_idle());
        drop(live);
    }

    #[test]
    fn a_live_run_needs_its_input_given_and_the_rest_held() {
        let message = |plan: Plan, input: &str| {
            Live::start(plan, compute(), input, |_| {})
                .unwrap_err()
                .to_string()
        };
        assert_eq!(
            message(gated(None).0, "frame"),
            "recipe `gated` takes input `offset`, which a live run must hold: it gives `frame` alone"
        );
        assert_eq!(
            message(gated(Some(0.0)).0, "other"),
            "recipe `gated` has no input `other` to run live"
        );
        let (mut held, _) = gated(Some(0.0));
        held.hold("frame", Value::Scalar(0.0)).unwrap();
        assert_eq!(
            message(held, "frame"),
            "recipe `gated` holds input `frame`; let it go to run it live"
        );
    }
}
