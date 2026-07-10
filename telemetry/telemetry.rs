use crossbeam_channel::{bounded, Receiver, Sender};
use parking_lot::RwLock;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

pub const TELEMETRY_RING_SIZE: usize = 512;
pub const TELEMETRY_SAMPLE_HZ: u32 = 1000;

#[derive(Clone, Copy, Debug)]
pub struct TelemetrySample {
    pub timestamp_ns: u64,
    pub edge_idx: u32,
    pub utilization: f32,
}

pub struct TelemetryRingBuffer {
    buffer: RwLock<[TelemetrySample; TELEMETRY_RING_SIZE]>,
    write_idx: AtomicU32,
}

impl TelemetryRingBuffer {
    pub fn new() -> Self {
        Self {
            buffer: RwLock::new([TelemetrySample {
                timestamp_ns: 0,
                edge_idx: 0,
                utilization: 0.0,
            }; TELEMETRY_RING_SIZE]),
            write_idx: AtomicU32::new(0),
        }
    }

    pub fn write(&self, edge_idx: u32, utilization: f32) {
        let idx = self.write_idx.fetch_add(1, Ordering::Relaxed) as usize % TELEMETRY_RING_SIZE;
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        let mut buf = self.buffer.write();
        buf[idx] = TelemetrySample {
            timestamp_ns: ts,
            edge_idx,
            utilization,
        };
    }

    pub fn snapshot(&self, num_edges: usize) -> Vec<f32> {
        let buf = self.buffer.read();
        let start = self.write_idx.load(Ordering::Acquire) as usize;
        let mut utils = vec![0.0f32; num_edges];
        for i in 0..TELEMETRY_RING_SIZE {
            let idx = (start + TELEMETRY_RING_SIZE - 1 - i) % TELEMETRY_RING_SIZE;
            let s = buf[idx];
            if (s.edge_idx as usize) < num_edges {
                if utils[s.edge_idx as usize] == 0.0 {
                    utils[s.edge_idx as usize] = s.utilization;
                }
            }
        }
        utils
    }
}

impl Default for TelemetryRingBuffer {
    fn default() -> Self {
        Self::new()
    }
}

pub struct TelemetryDaemon {
    ring: Arc<TelemetryRingBuffer>,
    running: Arc<AtomicBool>,
    simulated: Arc<RwLock<Vec<f32>>>,
    handle: Option<thread::JoinHandle<()>>,
    cmd_tx: Option<Sender<()>>,
}

impl TelemetryDaemon {
    pub fn new(num_edges: usize) -> Self {
        Self {
            ring: Arc::new(TelemetryRingBuffer::new()),
            running: Arc::new(AtomicBool::new(false)),
            simulated: Arc::new(RwLock::new(vec![0.0; num_edges])),
            handle: None,
            cmd_tx: None,
        }
    }

    pub fn ring(&self) -> Arc<TelemetryRingBuffer> {
        Arc::clone(&self.ring)
    }

    pub fn set_simulated_util(&self, edge_idx: u32, util: f32) {
        let mut sim = self.simulated.write();
        if (edge_idx as usize) < sim.len() {
            sim[edge_idx as usize] = util.clamp(0.0, 1.0);
        }
    }

    pub fn start(&mut self) {
        if self.running.swap(true, Ordering::SeqCst) {
            return;
        }
        let (tx, rx): (Sender<()>, Receiver<()>) = bounded(1);
        self.cmd_tx = Some(tx);

        let ring = Arc::clone(&self.ring);
        let running = Arc::clone(&self.running);
        let simulated = Arc::clone(&self.simulated);

        self.handle = Some(thread::spawn(move || {
            let interval = Duration::from_micros(1_000_000 / TELEMETRY_SAMPLE_HZ as u64);
            while running.load(Ordering::Relaxed) {
                let sim = simulated.read();
                for (e, &util) in sim.iter().enumerate() {
                    ring.write(e as u32, util);
                }
                drop(sim);
                thread::sleep(interval);
                if rx.try_recv().is_ok() {
                    break;
                }
            }
        }));
    }

    pub fn stop(&mut self) {
        self.running.store(false, Ordering::SeqCst);
        if let Some(tx) = self.cmd_tx.take() {
            let _ = tx.send(());
        }
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

impl Drop for TelemetryDaemon {
    fn drop(&mut self) {
        self.stop();
    }
}
