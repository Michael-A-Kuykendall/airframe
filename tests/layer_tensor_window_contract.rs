use airframe::backend::bindless::pipeline::{
    plan_layer_tensor_window, plan_layer_window, LayerOffsets,
};
use bytemuck::Zeroable;

#[test]
fn qwen17_layer7_ffn_up_tail_is_resident() {
    let chunk_words = 128 * 1024 * 1024 / 4;
    let base = 4 * chunk_words;
    let mut offsets = LayerOffsets::zeroed();
    let start = 670_935_456u64;
    offsets.ffn_up = ((start / 4 - u64::from(base)) * 2) as u32;
    let bytes = 2048 * 6144 / 256 * 144;
    let (old, _) = plan_layer_window(&offsets, base, chunk_words, 16).unwrap();
    assert!(
        old.unwrap()
            .absolute_to_local(((start + bytes - 1) / 4) as u32)
            .is_err(),
        "falsification: start-only planner must miss the matrix tail"
    );
    let (new, local_base) =
        plan_layer_tensor_window(&offsets, base, chunk_words, 16, &[(start, bytes)]).unwrap();
    let new = new.unwrap();
    assert_eq!(new.start_chunk, 4);
    assert_eq!(new.slot_count, 2);
    assert_eq!(local_base, 0);
    assert!(new
        .absolute_to_local(((start + bytes - 1) / 4) as u32)
        .is_ok());
}

#[test]
fn tensor_start_at_chunk_boundary_preserves_layer_base() {
    let mut offsets = LayerOffsets::zeroed();
    offsets.ffn_up = 2;
    let (window, local_base) = plan_layer_tensor_window(&offsets, 63, 64, 2, &[(256, 16)]).unwrap();
    let window = window.unwrap();
    let (slot, local) = window.absolute_to_local(64).unwrap();
    assert_eq!(local_base + offsets.ffn_up / 2, slot as u32 * 64 + local);
}

#[test]
fn exact_boundary_needs_no_extra_chunk() {
    let mut offsets = LayerOffsets::zeroed();
    offsets.ffn_up = 2;
    let (window, _) = plan_layer_tensor_window(&offsets, 0, 64, 1, &[(4, 252)]).unwrap();
    assert_eq!(window.unwrap().slot_count, 1);
}

#[test]
fn invalid_tensor_extents_fail_closed() {
    let mut offsets = LayerOffsets::zeroed();
    offsets.ffn_up = 2;
    for spans in [&[(4, 0)][..], &[(u64::MAX - 3, 8)][..], &[(4, 4096)][..]] {
        assert!(plan_layer_tensor_window(&offsets, 0, 64, 2, spans).is_err());
    }
}
