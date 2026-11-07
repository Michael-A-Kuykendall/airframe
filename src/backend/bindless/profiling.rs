//! Passive GPU profiling adapter. All schema, aggregation and I/O live in Observe.
use std::ops::{Deref, DerefMut};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};
use std::time::Instant;

#[cfg(feature = "isf")]
pub use airframe_observe::profiling::{KernelStage, ProfilePhase};
#[cfg(not(feature = "isf"))]
#[derive(Clone, Copy)]
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
#[cfg(not(feature = "isf"))]
#[allow(dead_code)]
#[derive(Clone, Copy)]
pub enum ProfilePhase {
    Prefill,
    Decode,
}

fn elapsed(start: Instant) -> u64 {
    start.elapsed().as_nanos().min(u64::MAX as u128) as u64
}
#[derive(Default)]
pub(crate) struct SubmissionTiming {
    overflow: std::sync::atomic::AtomicBool,
    submissions: AtomicU64,
    submit_ns: AtomicU64,
    wait_ns: AtomicU64,
    #[cfg_attr(not(feature = "isf"), allow(dead_code))]
    passes: AtomicU64,
    dispatches: AtomicU64,
}
impl SubmissionTiming {
    fn increment(&self, counter: &AtomicU64, value: u64) {
        let mut old = counter.load(Ordering::Relaxed);
        loop {
            let sum = old.checked_add(value);
            match counter.compare_exchange_weak(
                old,
                sum.unwrap_or(u64::MAX),
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    if sum.is_none() {
                        self.overflow.store(true, Ordering::Relaxed);
                    }
                    break;
                }
                Err(current) => old = current,
            }
        }
    }
    pub(crate) fn submit(&self, queue: &wgpu::Queue, command: wgpu::CommandBuffer) {
        let start = Instant::now();
        queue.submit(Some(command));
        self.increment(&self.submissions, 1);
        self.increment(&self.submit_ns, elapsed(start));
    }
    pub(crate) fn record_wait(&self, start: Instant) {
        self.increment(&self.wait_ns, elapsed(start));
    }
}
#[derive(Clone)]
pub(crate) struct ProfileContext {
    #[cfg(feature = "isf")]
    handle: airframe_observe::profiling::ProfileHandle,
    #[cfg(feature = "isf")]
    phase: ProfilePhase,
    #[cfg(feature = "isf")]
    pool: Arc<std::sync::Mutex<Option<super::pool_timestamp::TimestampPool>>>,
}
#[cfg(feature = "isf")]
pub(crate) fn generation_contexts(
    handle: airframe_observe::profiling::ProfileHandle,
) -> (ProfileContext, ProfileContext) {
    let pool = Arc::new(std::sync::Mutex::new(None));
    (
        ProfileContext {
            handle: handle.clone(),
            phase: ProfilePhase::Prefill,
            pool: pool.clone(),
        },
        ProfileContext {
            handle,
            phase: ProfilePhase::Decode,
            pool,
        },
    )
}
pub(crate) struct FrameProfile {
    counter: Option<Arc<SubmissionTiming>>,
    #[cfg(feature = "isf")]
    inner: Option<FrameInner>,
}
#[cfg(feature = "isf")]
struct PassRecord {
    sample: airframe_observe::profiling::StageMeasurement,
    query: Option<u32>,
}
#[cfg(feature = "isf")]
struct FrameInner {
    context: ProfileContext,
    pool: Option<super::pool_timestamp::TimestampPool>,
    records: Vec<PassRecord>,
    receiver: Option<std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>,
    started: Instant,
    setup_ns: Option<u64>,
    batch_size: u32,
    current_pos: u32,
    missing: airframe_observe::profiling::MissingGpu,
    contaminated: bool,
}
impl FrameProfile {
    pub(crate) fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        context: Option<ProfileContext>,
        batch_size: u32,
        current_pos: u32,
    ) -> Self {
        #[cfg(feature = "isf")]
        if let Some(context) = context {
            let started = Instant::now();
            let gpu = context.handle.gpu_requested();
            let mut pool = context.pool.lock().ok().and_then(|mut p| p.take());
            if gpu && pool.is_none() {
                pool = super::pool_timestamp::TimestampPool::for_compute_passes(
                    device,
                    wgpu::QUERY_SET_MAX_QUERIES / 2,
                    queue.get_timestamp_period() as f64,
                );
            }
            let missing = context.handle.missing_gpu_reason();
            context.handle.set_gpu_support(
                pool.is_some(),
                pool.as_ref().map(|_| queue.get_timestamp_period() as f64),
            );
            let contaminated = [
                "AIRFRAME_CAPTURE_INVARIANT",
                "AIRFRAME_PEEL",
                "AIRFRAME_TRACE_PREFILL_LAYERS",
                "AIRFRAME_LAYER_DUMP_CAPTURE",
            ]
            .iter()
            .any(|key| std::env::var(key).is_ok());
            return Self {
                counter: Some(Arc::new(SubmissionTiming::default())),
                inner: Some(FrameInner {
                    context,
                    pool,
                    records: Vec::new(),
                    receiver: None,
                    started,
                    setup_ns: None,
                    batch_size,
                    current_pos,
                    missing,
                    contaminated,
                }),
            };
        }
        let _ = (device, queue, context, batch_size, current_pos);
        Self {
            counter: None,
            #[cfg(feature = "isf")]
            inner: None,
        }
    }
    pub(crate) fn mark_contaminated(&mut self, active: bool) {
        #[cfg(feature = "isf")]
        if let Some(inner) = &mut self.inner {
            inner.contaminated |= active;
        }
        let _ = active;
    }

    pub(crate) fn submission_counter(&self) -> Option<Arc<SubmissionTiming>> {
        self.counter.clone()
    }
    pub(crate) fn begin_compute_pass<'a>(
        &'a mut self,
        encoder: &'a mut wgpu::CommandEncoder,
        descriptor: &wgpu::ComputePassDescriptor<'_>,
        layer: Option<u32>,
        stage: KernelStage,
    ) -> ProfiledPass<'a> {
        #[cfg(feature = "isf")]
        if let Some(inner) = self.inner.as_mut() {
            use airframe_observe::profiling::{MissingGpu, StageMeasurement};
            inner.setup_ns.get_or_insert_with(|| elapsed(inner.started));
            let started = Instant::now();
            let reservation = inner.pool.as_mut().and_then(|pool| pool.reserve_pass());
            let (query, timestamp_writes) = match reservation {
                Some((q, w)) => (Some(q), Some(w)),
                None => (None, None),
            };
            let pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: descriptor.label,
                timestamp_writes,
            });
            let record = if inner.records.len() < (wgpu::QUERY_SET_MAX_QUERIES / 2) as usize {
                inner.records.push(PassRecord {
                    query,
                    sample: StageMeasurement {
                        layer,
                        stage,
                        host_encode_ns: 0,
                        dispatches: 0,
                        gpu_ns: None,
                        missing: if inner.pool.is_some() && query.is_none() {
                            MissingGpu::Capacity
                        } else {
                            inner.missing
                        },
                    },
                });
                inner.records.last_mut()
            } else {
                None
            };
            if let Some(counter) = &self.counter {
                counter.increment(&counter.passes, 1);
            }
            return ProfiledPass {
                pass: Some(pass),
                started: Some(started),
                counter: self.counter.clone(),
                record,
            };
        }
        let _ = (layer, stage);
        ProfiledPass {
            pass: Some(encoder.begin_compute_pass(descriptor)),
            started: None,
            counter: None,
            #[cfg(feature = "isf")]
            record: None,
        }
    }
    pub(crate) fn resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        #[cfg(feature = "isf")]
        if let Some(pool) = self.inner.as_ref().and_then(|i| i.pool.as_ref()) {
            pool.resolve(encoder);
        }
        let _ = encoder;
    }
    pub(crate) fn submit_final(&self, queue: &wgpu::Queue, command: wgpu::CommandBuffer) {
        if let Some(counter) = &self.counter {
            counter.submit(queue, command);
        } else {
            queue.submit(Some(command));
        }
    }
    pub(crate) fn begin_readback(&mut self) {
        #[cfg(feature = "isf")]
        if let Some(inner) = &mut self.inner {
            if let Some(pool) = inner.pool.as_ref().filter(|p| p.used_pairs() > 0) {
                let (tx, rx) = std::sync::mpsc::channel();
                pool.readback_slice()
                    .map_async(wgpu::MapMode::Read, move |r| {
                        let _ = tx.send(r);
                    });
                inner.receiver = Some(rx);
            }
        }
    }
    pub(crate) fn wait_start(&self) -> Option<Instant> {
        self.counter.as_ref().map(|_| Instant::now())
    }
    pub(crate) fn wait_end(&self, start: Option<Instant>) {
        if let (Some(counter), Some(start)) = (&self.counter, start) {
            counter.record_wait(start);
        }
    }
    #[cfg(all(test, feature = "isf"))]
    #[allow(dead_code)] // Used by the standalone GPU integration harness, not lib tests.
    pub(crate) fn delay_readback_for_test(
        &mut self,
    ) -> std::sync::mpsc::Sender<Result<(), wgpu::BufferAsyncError>> {
        let (tx, rx) = std::sync::mpsc::channel();
        self.inner.as_mut().unwrap().receiver = Some(rx);
        tx
    }

    pub(crate) fn finish(&mut self, device: &wgpu::Device) {
        #[cfg(feature = "isf")]
        if let Some(mut inner) = self.inner.take() {
            use airframe_observe::profiling::MissingGpu;
            let forward_ns = elapsed(inner.started);
            let readback_start = Instant::now();
            let mut data = None;
            let mut missing = MissingGpu::ReadbackTimeout;
            if let (Some(pool), Some(rx)) = (&inner.pool, &inner.receiver) {
                loop {
                    match rx.try_recv() {
                        Ok(Ok(())) => {
                            let view = pool.readback_slice().get_mapped_range();
                            data = Some(bytemuck::cast_slice::<u8, u64>(&view).to_vec());
                            break;
                        }
                        Ok(Err(_)) | Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                            missing = MissingGpu::InvalidTimestamp;
                            break;
                        }
                        Err(std::sync::mpsc::TryRecvError::Empty) => {}
                    }
                    if readback_start.elapsed() >= std::time::Duration::from_millis(100) {
                        break;
                    }
                    if device.poll(wgpu::PollType::Poll).is_err() {
                        missing = MissingGpu::InvalidTimestamp;
                        break;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
                pool.unmap();
            }
            for record in &mut inner.records {
                if let Some(index) = record.query {
                    record.sample.gpu_ns = data
                        .as_deref()
                        .and_then(|d| inner.pool.as_ref()?.elapsed_ns(index, d))
                        .filter(|v| *v < u64::MAX as f64)
                        .map(|v| v.round() as u64);
                    if record.sample.gpu_ns.is_none() {
                        record.sample.missing = if data.is_some() {
                            MissingGpu::InvalidTimestamp
                        } else {
                            missing
                        };
                    }
                }
            }
            self.record(&mut inner, forward_ns, elapsed(readback_start));
            if let Some(mut pool) = inner.pool.take() {
                pool.reset();
                if let Ok(mut slot) = inner.context.pool.lock() {
                    *slot = Some(pool);
                }
            }
        }
        let _ = device;
    }
    #[cfg(feature = "isf")]
    fn record(&self, inner: &mut FrameInner, forward_ns: u64, readback_ns: u64) {
        use airframe_observe::profiling::{ForwardMeasurement, SubmissionStats};
        let Some(c) = &self.counter else {
            return;
        };
        let passes = c.passes.load(Ordering::Relaxed);
        inner.context.handle.record_forward(ForwardMeasurement {
            phase: inner.context.phase,
            batch_size: inner.batch_size,
            current_pos: inner.current_pos,
            forward_wall_ns: forward_ns,
            setup_wall_ns: inner.setup_ns.unwrap_or(forward_ns),
            profiling_readback_wall_ns: readback_ns,
            submissions: SubmissionStats {
                queue_submission_count: c.submissions.load(Ordering::Relaxed),
                queue_submit_ns_sum: c.submit_ns.load(Ordering::Relaxed),
                poll_wait_wall_ns_sum: c.wait_ns.load(Ordering::Relaxed),
            },
            compute_pass_count: passes,
            dispatch_count: c.dispatches.load(Ordering::Relaxed),
            counter_overflow: c.overflow.load(Ordering::Relaxed),
            dropped_passes: passes.saturating_sub(inner.records.len() as u64),
            stages: std::mem::take(&mut inner.records)
                .into_iter()
                .map(|r| r.sample)
                .collect(),
            contaminated: inner.contaminated,
        });
    }
}
impl Drop for FrameProfile {
    fn drop(&mut self) {
        #[cfg(feature = "isf")]
        if let Some(mut inner) = self.inner.take() {
            inner.context.handle.mark_error();
            if let Some(pool) = &inner.pool {
                if inner.receiver.is_some() {
                    pool.unmap();
                }
            }
            let wall = elapsed(inner.started);
            self.record(&mut inner, wall, 0);
        }
    }
}
pub(crate) struct ProfiledPass<'a> {
    pass: Option<wgpu::ComputePass<'a>>,
    #[cfg_attr(not(feature = "isf"), allow(dead_code))]
    started: Option<Instant>,
    counter: Option<Arc<SubmissionTiming>>,
    #[cfg(feature = "isf")]
    record: Option<&'a mut PassRecord>,
}
impl ProfiledPass<'_> {
    pub(crate) fn dispatch_workgroups(&mut self, x: u32, y: u32, z: u32) {
        self.pass
            .as_mut()
            .expect("live pass")
            .dispatch_workgroups(x, y, z);
        if let Some(counter) = &self.counter {
            counter.increment(&counter.dispatches, 1);
        }
        #[cfg(feature = "isf")]
        if let Some(record) = &mut self.record {
            record.sample.dispatches = record.sample.dispatches.saturating_add(1);
        }
    }
}
impl<'a> Deref for ProfiledPass<'a> {
    type Target = wgpu::ComputePass<'a>;
    fn deref(&self) -> &Self::Target {
        self.pass.as_ref().expect("live pass")
    }
}
impl DerefMut for ProfiledPass<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.pass.as_mut().expect("live pass")
    }
}
impl Drop for ProfiledPass<'_> {
    fn drop(&mut self) {
        self.pass.take();
        #[cfg(feature = "isf")]
        if let (Some(start), Some(record)) = (self.started, &mut self.record) {
            record.sample.host_encode_ns = elapsed(start);
        }
    }
}
