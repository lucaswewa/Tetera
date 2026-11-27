//! The frame model at work: a pool of 16 full MIChrome 20 frames (about 640 MB) feeds 48
//! frames to three consumers, which share each one without copying. Then the pool runs dry.
//!
//! Run it with `cargo run -p tetera-core --example frame_pool`; `-- --smoke` checks the
//! pool's counts, and says so.

use std::sync::mpsc;
use std::time::Duration;
use tetera_core::frame::{BayerPattern, HostAllocator};
use tetera_core::{
    Frame, FrameLayout, FrameMut, FramePool, PixelFormat, PixelView, Result, Timestamp,
};

const FRAMES: u64 = 48;
const CONSUMERS: usize = 3;

fn main() -> Result<()> {
    let smoke = std::env::args().any(|arg| arg == "--smoke");
    let layout = FrameLayout::new(PixelFormat::bayer16(BayerPattern::Rggb, 12)?, 5_472, 3_648)?;
    let pool = FramePool::new(layout, 16, &HostAllocator)?;
    println!(
        "A pool of 16 frames of 5472 × 3648 Bayer16 (12 bits): {:.1} MB each, {:.0} MB in all.\n",
        layout.bytes() as f64 / 1e6,
        16.0 * layout.bytes() as f64 / 1e6
    );

    std::thread::scope(|s| -> Result<()> {
        let (done, results) = mpsc::channel::<(usize, u64, f64)>();
        let mut consumers = Vec::new();
        for c in 0..CONSUMERS {
            let (send, frames) = mpsc::channel::<Frame>();
            consumers.push(send);
            let done = done.clone();
            s.spawn(move || {
                // Each reads one row's mean, then drops its share of the frame.
                for frame in frames {
                    let row = frame.samples_u16(c as u32 * 1_000);
                    let mean = row.iter().map(|&v| f64::from(v)).sum::<f64>() / row.len() as f64;
                    done.send((c, frame.meta().seq, mean))
                        .expect("main is listening");
                }
            });
        }
        drop(done);

        for seq in 0..FRAMES {
            let mut frame = pool.take(Duration::from_secs(10))?;
            frame.data_mut().fill((seq % 16) as u8); // every sample becomes level × 257
            let meta = frame.meta_mut();
            meta.seq = seq;
            meta.exposure_start = Timestamp::from_duration(Duration::from_millis(seq * 138));
            meta.exposure_end = Timestamp::from_duration(Duration::from_millis(seq * 138 + 12));
            let frame = frame.freeze();
            for consumer in &consumers {
                consumer
                    .send(frame.clone())
                    .expect("the consumer is running");
            }
        }
        drop(consumers);

        for (c, seq, mean) in results {
            if seq % 12 == 0 {
                println!("consumer {c} read frame {seq:>2}: its row's mean is {mean:>6.0}");
            }
        }
        Ok(())
    })?;

    let stats = pool.stats();
    println!(
        "\n{FRAMES} frames through 16 buffers: {} taken, {} allocated, {} takes waited for a buffer to come back.",
        stats.takes, stats.allocations, stats.waits
    );

    let held: Vec<FrameMut> = (0..16)
        .map(|_| pool.take(Duration::ZERO))
        .collect::<Result<_>>()?;
    match pool.take(Duration::from_millis(200)) {
        Err(e) => println!("With all 16 lent, a 17th: {e}."),
        Ok(_) => println!("With all 16 lent, a 17th was lent too: that's a fault."),
    }
    drop(held);
    let after = pool.take(Duration::ZERO).map(|_| "lent at once")?;
    println!("Once they come back, a frame is {after}.");

    if smoke {
        let stats = pool.stats();
        assert_eq!(stats.allocations, 16, "a warm pool allocates nothing more");
        assert_eq!(stats.takes, FRAMES + 16 + 1);
        assert_eq!((stats.misses, stats.in_use), (1, 0));
        println!("\nsmoke: ok");
    }
    Ok(())
}
