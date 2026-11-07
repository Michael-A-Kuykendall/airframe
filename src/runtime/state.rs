//! Request-isolated state contracts. Stored bytes are never evidence of initialized state.
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc,
};
use thiserror::Error;
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StateIdentity {
    pub model_sha256: String,
    pub plan_sha256: String,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvPrecision {
    F32,
    F32WithInt4Shadow,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StateKind {
    DenseKv,
    WindowedKv,
    LatentKv,
    DeltaNet,
    Mamba,
    Rwkv,
    EncoderMemory,
}
#[derive(Clone, Debug)]
pub struct StateSpec {
    pub kind: StateKind,
    pub layers: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub capacity: usize,
    pub precision: KvPrecision,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StateBudget {
    pub per_tensor_bytes: u64,
    pub total_bytes: u64,
}
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum StateError {
    #[error("unsupported_state: {0}")]
    Unsupported(String),
    #[error("state_bounds: {0}")]
    Bounds(String),
    #[error("state_identity_mismatch")]
    Identity,
    #[error("state_busy")]
    Busy,
    #[error("state_phase: {0}")]
    Phase(String),
    #[error("state_epoch_exhausted")]
    Epoch,
}
type Result<T> = std::result::Result<T, StateError>;
impl StateSpec {
    pub fn budget(&self, max_buffer: u64, max_total: u64) -> Result<StateBudget> {
        if self.kind != StateKind::DenseKv {
            return Err(StateError::Unsupported(format!("{:?}", self.kind)));
        }
        if [self.layers, self.kv_heads, self.head_dim, self.capacity].contains(&0) {
            return Err(StateError::Bounds("zero geometry".into()));
        }
        let elements = [self.capacity, self.kv_heads, self.head_dim]
            .iter()
            .try_fold(1u64, |a, b| a.checked_mul(*b as u64))
            .ok_or_else(|| StateError::Bounds("KV product overflow".into()))?;
        let bytes = elements
            .checked_mul(4)
            .ok_or_else(|| StateError::Bounds("KV byte overflow".into()))?;
        let mut per_layer = bytes
            .checked_mul(2)
            .ok_or_else(|| StateError::Bounds("KV byte overflow".into()))?;
        if self.precision == KvPrecision::F32WithInt4Shadow {
            if !self.head_dim.is_multiple_of(8) {
                return Err(StateError::Unsupported(
                    "INT4 head dimension must divide 8".into(),
                ));
            }
            let packed = elements / 8 * 4;
            let scales = (self.capacity as u64)
                .checked_mul(self.kv_heads as u64)
                .and_then(|x| x.checked_mul(4))
                .ok_or_else(|| StateError::Bounds("scale byte overflow".into()))?;
            per_layer = per_layer
                .checked_add(
                    packed
                        .checked_mul(2)
                        .ok_or_else(|| StateError::Bounds("packed bytes".into()))?,
                )
                .and_then(|x| x.checked_add(scales.checked_mul(2)?))
                .ok_or_else(|| StateError::Bounds("shadow byte overflow".into()))?;
        }
        let total = per_layer
            .checked_mul(self.layers as u64)
            .ok_or_else(|| StateError::Bounds("layer byte overflow".into()))?;
        if bytes > max_buffer || total > max_total {
            return Err(StateError::Bounds(format!(
                "KV budget per-buffer={bytes}, total={total}"
            )));
        }
        Ok(StateBudget {
            per_tensor_bytes: bytes,
            total_bytes: total,
        })
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Fresh,
    Prefill,
    Ready,
    Decode,
    Finished,
    Aborted,
}
#[derive(Clone, Debug)]
pub struct StateSnapshot {
    pub epoch: u64,
    pub phase: Phase,
    pub processed_positions: usize,
    pub prompt_positions: usize,
}
pub struct StateController {
    identity: StateIdentity,
    spec: StateSpec,
    epoch: AtomicU64,
    active: AtomicBool,
}
impl StateController {
    pub fn new(identity: StateIdentity, spec: StateSpec) -> Result<Arc<Self>> {
        spec.budget(u64::MAX, u64::MAX)?;
        Ok(Arc::new(Self {
            identity,
            spec,
            epoch: AtomicU64::new(0),
            active: AtomicBool::new(false),
        }))
    }
    #[allow(deprecated)]
    pub fn begin(self: &Arc<Self>) -> Result<StateLease> {
        if self
            .active
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Err(StateError::Busy);
        }
        let epoch = match self
            .epoch
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |v| v.checked_add(1))
        {
            Ok(v) => v + 1,
            Err(_) => {
                self.active.store(false, Ordering::Release);
                return Err(StateError::Epoch);
            }
        };
        Ok(StateLease {
            owner: self.clone(),
            epoch,
            phase: Phase::Fresh,
            processed: 0,
            prompt: 0,
            pending: None,
            closed: false,
        })
    }
    pub fn identity(&self) -> &StateIdentity {
        &self.identity
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StateWrite {
    epoch: u64,
    start: usize,
    count: usize,
    phase: Phase,
}
pub struct StateLease {
    owner: Arc<StateController>,
    epoch: u64,
    phase: Phase,
    processed: usize,
    prompt: usize,
    pending: Option<StateWrite>,
    closed: bool,
}
impl StateLease {
    pub fn snapshot(&self) -> StateSnapshot {
        StateSnapshot {
            epoch: self.epoch,
            phase: self.phase,
            processed_positions: self.processed,
            prompt_positions: self.prompt,
        }
    }
    pub fn check_identity(&self, expected: &StateIdentity) -> Result<()> {
        if &self.owner.identity != expected
            || self.owner.epoch.load(Ordering::Acquire) != self.epoch
            || self.closed
        {
            return Err(StateError::Identity);
        }
        Ok(())
    }
    pub fn start_prefill(&mut self, tokens: usize) -> Result<()> {
        if self.phase != Phase::Fresh || tokens == 0 || tokens > self.owner.spec.capacity {
            return Err(StateError::Phase(
                "prefill requires fresh lease and fitting nonempty prompt".into(),
            ));
        }
        self.prompt = tokens;
        self.phase = Phase::Prefill;
        Ok(())
    }
    pub fn prepare_prefill_chunk(&mut self, start: usize, count: usize) -> Result<StateWrite> {
        if self.phase != Phase::Prefill {
            return Err(StateError::Phase("not prefilling".into()));
        }
        self.prepare(start, count, self.prompt)
    }
    pub fn prepare_decode(&mut self, start: usize) -> Result<StateWrite> {
        if self.phase != Phase::Ready {
            return Err(StateError::Phase("decode requires complete prefill".into()));
        }
        self.phase = Phase::Decode;
        match self.prepare(start, 1, self.owner.spec.capacity) {
            Ok(w) => Ok(w),
            Err(e) => {
                self.phase = Phase::Ready;
                Err(e)
            }
        }
    }
    fn prepare(&mut self, start: usize, count: usize, limit: usize) -> Result<StateWrite> {
        if self.closed
            || self.pending.is_some()
            || count == 0
            || start != self.processed
            || start
                .checked_add(count)
                .is_none_or(|n| n > limit || n > self.owner.spec.capacity)
        {
            return Err(StateError::Bounds(
                "write must extend exactly the committed prefix within capacity".into(),
            ));
        }
        let w = StateWrite {
            epoch: self.epoch,
            start,
            count,
            phase: self.phase,
        };
        self.pending = Some(w);
        Ok(w)
    }
    pub fn commit(&mut self, write: StateWrite) -> Result<()> {
        if self.pending != Some(write) || write.epoch != self.epoch || self.closed {
            return Err(StateError::Identity);
        }
        self.processed = write.start + write.count;
        self.pending = None;
        if self.phase == Phase::Decode || self.processed == self.prompt {
            self.phase = Phase::Ready;
        }
        Ok(())
    }
    pub fn require_read(&self, start: usize, end: usize) -> Result<()> {
        if self.closed || start > end || end > self.processed {
            return Err(StateError::Bounds("read outside committed prefix".into()));
        }
        Ok(())
    }
    pub fn finish(&mut self) -> Result<()> {
        if self.phase != Phase::Ready || self.pending.is_some() {
            return Err(StateError::Phase(
                "cannot finish incomplete/failed state".into(),
            ));
        }
        self.close(Phase::Finished);
        Ok(())
    }
    pub fn abort(&mut self) {
        self.close(Phase::Aborted);
    }
    fn close(&mut self, phase: Phase) {
        if self.closed {
            return;
        }
        self.phase = phase;
        self.processed = 0;
        self.pending = None;
        self.closed = true;
        self.owner.active.store(false, Ordering::Release);
    }
    pub fn rollback(&mut self, _positions: usize) -> Result<()> {
        Err(StateError::Unsupported(
            "prefix rollback/checkpoint reuse is not certified".into(),
        ))
    }
}
impl Drop for StateLease {
    fn drop(&mut self) {
        if !self.closed {
            self.abort();
        }
    }
}
