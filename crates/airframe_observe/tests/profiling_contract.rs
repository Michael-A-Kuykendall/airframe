use airframe_observe::profiling::*;
use serde_json::Value;
use std::io::Write;
use std::sync::{Arc, Mutex};

#[derive(Clone, Default)]
struct Capture(Arc<Mutex<Vec<u8>>>);
impl Write for Capture {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
impl Capture {
    fn lines(&self) -> Vec<Value> {
        String::from_utf8(self.0.lock().unwrap().clone())
            .unwrap()
            .lines()
            .map(|s| serde_json::from_str(s).unwrap())
            .collect()
    }
}
fn frame(phase: ProfilePhase, layer: u32, gpu: Option<u64>, dispatches: u64) -> ForwardMeasurement {
    ForwardMeasurement {
        phase,
        batch_size: 1,
        current_pos: 0,
        forward_wall_ns: 50,
        setup_wall_ns: 2,
        profiling_readback_wall_ns: 3,
        submissions: SubmissionStats {
            queue_submission_count: 1,
            queue_submit_ns_sum: 5,
            poll_wait_wall_ns_sum: 40,
        },
        compute_pass_count: 1,
        dispatch_count: dispatches,
        counter_overflow: false,
        dropped_passes: 0,
        contaminated: false,
        stages: vec![StageMeasurement {
            layer: Some(layer),
            stage: KernelStage::Qkv,
            host_encode_ns: 4,
            dispatches,
            gpu_ns: gpu,
            missing: MissingGpu::InvalidTimestamp,
        }],
    }
}
#[test]
fn scalar_schema_phase_counts_and_partial_gpu_are_truthful() {
    let out = Capture::default();
    let config = ProfileConfig::with_writer(Box::new(out.clone()), true);
    config.set_adapter("test GPU".into(), "Vulkan".into());
    let mut session = config.begin_generation();
    let handle = session.handle();
    handle.set_prompt_tokens(1);
    handle.set_gpu_support(true, Some(1.0));
    handle.record_forward(frame(ProfilePhase::Prefill, 0, Some(19), 1));
    handle.record_forward(frame(ProfilePhase::Decode, 0, None, 2));
    session.finish(true);
    session.finish(false);
    drop(session);
    let lines = out.lines();
    assert_eq!(lines.len(), 2);
    let v = &lines[1];
    assert_eq!(v["schema"], "airframe.profile.v1");
    assert_eq!(v["status"], "complete");
    assert_eq!(v["prefill_forwards"], 1);
    assert_eq!(v["decode_forwards"], 1);
    assert_eq!(v["compute_pass_count"], 2);
    assert_eq!(v["dispatch_count"], 3);
    assert_eq!(v["submissions"]["queue_submission_count"], 2);
    assert_eq!(v["stages"][0]["phase"], "prefill");
    assert_eq!(v["stages"][0]["totals"]["gpu_ns_sum"], 19);
    assert!(v["stages"][1]["totals"]["gpu_ns_sum"].is_null());
    assert_eq!(
        v["stages"][1]["totals"]["missing_gpu"]["invalid_timestamp"],
        1
    );
    for forbidden in ["prompt", "tokens", "weights", "model_path", "text"] {
        assert!(v.get(forbidden).is_none());
    }
}
#[test]
fn repeated_calls_aggregate_without_retaining_tokens() {
    let out = Capture::default();
    let mut session = ProfileConfig::with_writer(Box::new(out.clone()), false).begin_generation();
    for _ in 0..5000 {
        session
            .handle()
            .record_forward(frame(ProfilePhase::Decode, 0, None, 1));
    }
    session.finish(true);
    let lines = out.lines();
    assert_eq!(lines.len(), 2);
    assert_eq!(lines[1]["stages"].as_array().unwrap().len(), 1);
    assert_eq!(lines[1]["stages"][0]["totals"]["calls"], 5000);
}
#[test]
fn aggregate_capacity_and_counter_overflow_are_reported() {
    let out = Capture::default();
    let mut session = ProfileConfig::with_writer(Box::new(out.clone()), false).begin_generation();
    for i in 0..4100 {
        session
            .handle()
            .record_forward(frame(ProfilePhase::Decode, i, None, 1));
    }
    session
        .handle()
        .record_forward(frame(ProfilePhase::Decode, 0, None, u64::MAX));
    session.finish(true);
    let lines = out.lines();
    assert_eq!(lines[1]["stages"].as_array().unwrap().len(), 4096);
    assert_eq!(lines[1]["dropped_measurements"], 4);
    assert_eq!(lines[1]["counter_overflow"], true);
    assert_eq!(lines[1]["dispatch_count"].as_u64(), Some(u64::MAX));
}
#[test]
fn terminal_error_and_unwind_emit_once() {
    let out = Capture::default();
    let config = ProfileConfig::with_writer(Box::new(out.clone()), false);
    {
        let mut session = config.begin_generation();
        session.handle().mark_error();
        session.finish(true);
    }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _session = config.begin_generation();
        panic!("test cancellation");
    }));
    assert!(result.is_err());
    let lines = out.lines();
    assert_eq!(lines.len(), 4);
    assert_eq!(lines[1]["status"], "error");
    assert_eq!(lines[3]["status"], "aborted");
    assert_ne!(lines[0]["generation_id"], lines[2]["generation_id"]);
}
#[test]
fn load_and_generation_use_distinct_events() {
    let out = Capture::default();
    let config = ProfileConfig::with_writer(Box::new(out.clone()), false);
    {
        let mut load = config.begin_load();
        load.complete();
    }
    {
        let _load = config.begin_load();
    }
    let lines = out.lines();
    assert_eq!(lines[0]["event"], "load_start");
    assert_eq!(lines[1]["event"], "load_end");
    assert_eq!(lines[1]["status"], "complete");
    assert_eq!(lines[3]["status"], "error");
}
#[test]
fn concurrent_generations_do_not_mix_or_interleave() {
    let out = Capture::default();
    let config = ProfileConfig::with_writer(Box::new(out.clone()), false);
    let jobs: Vec<_> = (0..4)
        .map(|i| {
            let config = config.clone();
            std::thread::spawn(move || {
                let mut s = config.begin_generation();
                s.handle().set_prompt_tokens(i + 1);
                s.finish(true);
            })
        })
        .collect();
    for j in jobs {
        j.join().unwrap();
    }
    let lines = out.lines();
    assert_eq!(lines.len(), 8);
    let ids: std::collections::BTreeSet<_> = lines
        .iter()
        .filter(|v| v["event"] == "generation_summary")
        .map(|v| v["generation_id"].as_u64().unwrap())
        .collect();
    assert_eq!(ids.len(), 4);
}
#[test]
fn profiling_is_off_by_default() {
    let saved = std::env::var_os("AIRFRAME_TRACE_TIMING");
    std::env::remove_var("AIRFRAME_TRACE_TIMING");
    assert!(ProfileConfig::from_env().is_none());
    if let Some(value) = saved {
        std::env::set_var("AIRFRAME_TRACE_TIMING", value);
    }
}
