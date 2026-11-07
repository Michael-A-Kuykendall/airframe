//! Scalar-only inference profiling. GPU resources and scheduling stay in Airframe.
use serde::Serialize;
use std::collections::BTreeMap;
use std::io::{self, Write};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, Mutex,
};
use std::time::Instant;

static NEXT_ID: AtomicU64 = AtomicU64::new(1);
const MAX_KEYS: usize = 4096;
fn ns(start: Instant) -> u64 {
    start.elapsed().as_nanos().min(u64::MAX as u128) as u64
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ProfilePhase {
    Prefill,
    Decode,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum KernelStage {
    AttnNorm,
    Qkv,
    QkNorm,
    AttnOut,
    AttnProj,
    PostAttnNorm,
    FfnNorm,
    FfnProj,
    FfnDown,
    PostFfnNorm,
    FinalNorm,
    LmHead,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MissingGpu {
    Unsupported,
    DeviceFeatureRequestFailed,
    HostOnly,
    Capacity,
    InvalidTimestamp,
    ReadbackTimeout,
}
#[derive(Clone, Debug)]
pub struct StageMeasurement {
    pub layer: Option<u32>,
    pub stage: KernelStage,
    pub host_encode_ns: u64,
    pub dispatches: u64,
    pub gpu_ns: Option<u64>,
    pub missing: MissingGpu,
}
#[derive(Default, Clone, Debug, Serialize)]
pub struct SubmissionStats {
    pub queue_submission_count: u64,
    pub queue_submit_ns_sum: u64,
    pub poll_wait_wall_ns_sum: u64,
}
#[derive(Debug)]
pub struct ForwardMeasurement {
    pub phase: ProfilePhase,
    pub compute_pass_count: u64,
    pub dispatch_count: u64,
    pub counter_overflow: bool,
    pub batch_size: u32,
    pub current_pos: u32,
    pub forward_wall_ns: u64,
    pub setup_wall_ns: u64,
    pub profiling_readback_wall_ns: u64,
    pub submissions: SubmissionStats,
    pub stages: Vec<StageMeasurement>,
    pub dropped_passes: u64,
    pub contaminated: bool,
}
#[derive(Clone)]
pub struct ProfileConfig {
    inner: Arc<Config>,
}
struct Config {
    gpu_failure: Mutex<Option<MissingGpu>>,
    runtime_id: u64,
    sink: Mutex<Box<dyn Write + Send>>,
    gpu: bool,
    adapter: Mutex<(String, String)>,
}
impl ProfileConfig {
    pub fn set_gpu_unavailable(&self, reason: MissingGpu) {
        if let Ok(mut slot) = self.inner.gpu_failure.lock() {
            *slot = Some(reason);
        }
    }

    pub fn from_env() -> Option<Self> {
        if std::env::var("AIRFRAME_TRACE_TIMING").ok().as_deref() != Some("1") {
            return None;
        }
        let sink: Box<dyn Write + Send> = match std::env::var_os("AIRFRAME_TRACE_TIMING_PATH") {
            Some(path) => match std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
            {
                Ok(file) => Box::new(file),
                Err(_) => {
                    eprintln!("[Airframe] profiling sink unavailable; using stderr");
                    Box::new(io::stderr())
                }
            },
            None => Box::new(io::stderr()),
        };
        Some(Self::with_writer(
            sink,
            std::env::var("AIRFRAME_TRACE_TIMING_GPU").ok().as_deref() != Some("0"),
        ))
    }
    pub fn with_writer(sink: Box<dyn Write + Send>, gpu: bool) -> Self {
        Self {
            inner: Arc::new(Config {
                gpu_failure: Mutex::new(None),
                runtime_id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
                sink: Mutex::new(sink),
                gpu,
                adapter: Mutex::new((String::new(), String::new())),
            }),
        }
    }
    pub fn gpu_requested(&self) -> bool {
        self.inner.gpu
    }
    pub fn set_adapter(&self, name: String, backend: String) {
        if let Ok(mut info) = self.inner.adapter.lock() {
            *info = (name, backend);
        }
    }
    fn write(&self, value: serde_json::Value) {
        let Ok(mut line) = serde_json::to_vec(&value) else {
            return;
        };
        line.push(b'\n');
        // One complete write protected across all runtime writers, including append files.
        static WRITE_LOCK: Mutex<()> = Mutex::new(());
        let Ok(_global) = WRITE_LOCK.lock() else {
            return;
        };
        let result = self
            .inner
            .sink
            .lock()
            .ok()
            .map(|mut w| w.write_all(&line).and_then(|_| w.flush()));
        if !matches!(result, Some(Ok(()))) {
            let _ = io::stderr().write_all(&line);
        }
    }
    pub fn begin_load(&self) -> LoadProfile {
        self.write(serde_json::json!({"schema":"airframe.profile.v1","event":"load_start","runtime_id":self.inner.runtime_id}));
        LoadProfile {
            config: self.clone(),
            started: Instant::now(),
            complete: false,
        }
    }
    pub fn begin_generation(&self) -> ProfileSession {
        let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        self.write(
            serde_json::json!({"schema":"airframe.profile.v1","event":"generation_start",
            "runtime_id":self.inner.runtime_id,"generation_id":id}),
        );
        ProfileSession {
            handle: ProfileHandle {
                config: self.clone(),
                data: Arc::new(Mutex::new(Data::default())),
            },
            id,
            started: Instant::now(),
            finished: false,
        }
    }
}
pub struct LoadProfile {
    config: ProfileConfig,
    started: Instant,
    complete: bool,
}
impl LoadProfile {
    pub fn complete(&mut self) {
        self.complete = true;
    }
}
impl Drop for LoadProfile {
    fn drop(&mut self) {
        self.config.write(
            serde_json::json!({"schema":"airframe.profile.v1","event":"load_end",
            "runtime_id":self.config.inner.runtime_id,"elapsed_ns":ns(self.started),
            "status":if self.complete {"complete"} else {"error"}}),
        );
    }
}
#[derive(Default, Serialize)]
struct Aggregate {
    calls: u64,
    compute_passes: u64,
    dispatches: u64,
    host_encode_ns: u64,
    gpu_ns_sum: Option<u64>,
    gpu_measured_calls: u64,
    missing_gpu: BTreeMap<MissingGpu, u64>,
}
#[derive(Default)]
struct Data {
    gpu_timestamp_period_ns: Option<f64>,
    gpu_timestamp_enabled: bool,
    min_batch_size: u32,
    max_batch_size: u32,
    max_current_pos: u32,
    max_seq_len: u64,
    prefill_chunk: u32,
    stages: BTreeMap<(ProfilePhase, Option<u32>, KernelStage), Aggregate>,
    forward_count: u64,
    prefill_forwards: u64,
    decode_forwards: u64,
    prompt_tokens: usize,
    forward_wall_ns_sum: u64,
    setup_wall_ns_sum: u64,
    profiling_readback_wall_ns_sum: u64,
    submissions: SubmissionStats,
    compute_pass_count: u64,
    dispatch_count: u64,
    dropped: u64,
    overflow: bool,
    error: bool,
    contaminated: bool,
}
fn add(target: &mut u64, value: u64, overflow: &mut bool) {
    match target.checked_add(value) {
        Some(v) => *target = v,
        None => {
            *target = u64::MAX;
            *overflow = true;
        }
    }
}
#[derive(Clone)]
pub struct ProfileHandle {
    config: ProfileConfig,
    data: Arc<Mutex<Data>>,
}
impl ProfileHandle {
    pub fn missing_gpu_reason(&self) -> MissingGpu {
        if !self.gpu_requested() {
            return MissingGpu::HostOnly;
        }
        self.config
            .inner
            .gpu_failure
            .lock()
            .ok()
            .and_then(|v| *v)
            .unwrap_or(MissingGpu::Unsupported)
    }

    pub fn set_gpu_support(&self, enabled: bool, period_ns: Option<f64>) {
        if let Ok(mut d) = self.data.lock() {
            d.gpu_timestamp_enabled = enabled;
            d.gpu_timestamp_period_ns = period_ns;
        }
    }
    pub fn gpu_requested(&self) -> bool {
        self.config.gpu_requested()
    }
    pub fn mark_error(&self) {
        if let Ok(mut d) = self.data.lock() {
            d.error = true;
        }
    }
    pub fn set_prefill_chunk(&self, chunk: u32) {
        if let Ok(mut data) = self.data.lock() {
            data.prefill_chunk = chunk;
        }
    }

    pub fn set_prompt_tokens(&self, count: usize) {
        if let Ok(mut d) = self.data.lock() {
            d.prompt_tokens = count;
        }
    }
    pub fn record_forward(&self, frame: ForwardMeasurement) {
        let Ok(mut d) = self.data.lock() else {
            return;
        };
        let mut overflow = d.overflow || frame.counter_overflow;
        d.min_batch_size = if d.forward_count == 0 {
            frame.batch_size
        } else {
            d.min_batch_size.min(frame.batch_size)
        };
        d.max_batch_size = d.max_batch_size.max(frame.batch_size);
        d.max_current_pos = d.max_current_pos.max(frame.current_pos);
        d.max_seq_len = d
            .max_seq_len
            .max(u64::from(frame.current_pos) + u64::from(frame.batch_size));
        add(
            &mut d.compute_pass_count,
            frame.compute_pass_count,
            &mut overflow,
        );
        add(&mut d.dispatch_count, frame.dispatch_count, &mut overflow);
        add(&mut d.forward_count, 1, &mut overflow);
        match frame.phase {
            ProfilePhase::Prefill => add(&mut d.prefill_forwards, 1, &mut overflow),
            ProfilePhase::Decode => add(&mut d.decode_forwards, 1, &mut overflow),
        }
        add(
            &mut d.forward_wall_ns_sum,
            frame.forward_wall_ns,
            &mut overflow,
        );
        add(&mut d.setup_wall_ns_sum, frame.setup_wall_ns, &mut overflow);
        add(
            &mut d.profiling_readback_wall_ns_sum,
            frame.profiling_readback_wall_ns,
            &mut overflow,
        );
        add(
            &mut d.submissions.queue_submission_count,
            frame.submissions.queue_submission_count,
            &mut overflow,
        );
        add(
            &mut d.submissions.queue_submit_ns_sum,
            frame.submissions.queue_submit_ns_sum,
            &mut overflow,
        );
        add(
            &mut d.submissions.poll_wait_wall_ns_sum,
            frame.submissions.poll_wait_wall_ns_sum,
            &mut overflow,
        );
        add(&mut d.dropped, frame.dropped_passes, &mut overflow);
        d.contaminated |= frame.contaminated;
        for sample in frame.stages {
            let key = (frame.phase, sample.layer, sample.stage);
            if d.stages.len() >= MAX_KEYS && !d.stages.contains_key(&key) {
                add(&mut d.dropped, 1, &mut overflow);
                continue;
            }
            let agg = d.stages.entry(key).or_default();
            add(&mut agg.calls, 1, &mut overflow);
            add(&mut agg.compute_passes, 1, &mut overflow);
            add(&mut agg.dispatches, sample.dispatches, &mut overflow);
            add(
                &mut agg.host_encode_ns,
                sample.host_encode_ns,
                &mut overflow,
            );
            if let Some(gpu) = sample.gpu_ns {
                add(agg.gpu_ns_sum.get_or_insert(0), gpu, &mut overflow);
                add(&mut agg.gpu_measured_calls, 1, &mut overflow);
            } else {
                add(
                    agg.missing_gpu.entry(sample.missing).or_default(),
                    1,
                    &mut overflow,
                );
            }
        }
        d.overflow = overflow;
    }
}
pub struct ProfileSession {
    handle: ProfileHandle,
    id: u64,
    started: Instant,
    finished: bool,
}
impl ProfileSession {
    pub fn handle(&self) -> ProfileHandle {
        self.handle.clone()
    }
    pub fn finish(&mut self, success: bool) {
        if self.finished {
            return;
        }
        self.finished = true;
        self.emit(if success { "complete" } else { "error" });
    }
    fn emit(&self, status: &str) {
        let Ok(d) = self.handle.data.lock() else {
            return;
        };
        let (adapter, backend) = self
            .handle
            .config
            .inner
            .adapter
            .lock()
            .map(|v| v.clone())
            .unwrap_or_default();
        let stages: Vec<_> = d
            .stages
            .iter()
            .map(|((phase, layer, stage), totals)| {
                serde_json::json!({
                    "phase":phase,"layer":layer,"stage":stage,"totals":totals
                })
            })
            .collect();
        self.handle.config.write(serde_json::json!({
            "schema":"airframe.profile.v1","event":"generation_summary",
            "runtime_id":self.handle.config.inner.runtime_id,"generation_id":self.id,
            "status":if d.error {"error"} else {status}, "generation_wall_ns":ns(self.started),
            "adapter":adapter,"backend":backend,"prompt_tokens":d.prompt_tokens,
            "forward_count":d.forward_count,"prefill_forwards":d.prefill_forwards,"decode_forwards":d.decode_forwards,
            "forward_wall_ns_sum":d.forward_wall_ns_sum,"setup_wall_ns_sum":d.setup_wall_ns_sum,
            "profiling_readback_wall_ns_sum":d.profiling_readback_wall_ns_sum,
            "submissions":d.submissions,"compute_pass_count":d.compute_pass_count,"dispatch_count":d.dispatch_count,
            "dropped_measurements":d.dropped,"counter_overflow":d.overflow,"contaminated":d.contaminated,
            "gpu_timestamp_period_ns":d.gpu_timestamp_period_ns,"gpu_timestamp_enabled":d.gpu_timestamp_enabled,
            "min_batch_size":d.min_batch_size,"max_batch_size":d.max_batch_size,"max_current_pos":d.max_current_pos,
            "max_seq_len":d.max_seq_len,"prefill_chunk":d.prefill_chunk,
            "stages":stages
        }));
    }
}
impl Drop for ProfileSession {
    fn drop(&mut self) {
        if !self.finished {
            self.finished = true;
            self.emit("aborted");
        }
    }
}
