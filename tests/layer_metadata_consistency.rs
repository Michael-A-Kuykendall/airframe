use airframe::backend::bindless::pipeline::{plan_layer_tensor_window, LayerOffsets};
use bytemuck::Zeroable;

#[test]
fn descriptor_tensors_without_offsets_are_rejected() {
    let offsets = LayerOffsets::zeroed();
    assert!(plan_layer_tensor_window(&offsets, 0, 64, 2, &[(4, 16)]).is_err());
}

#[test]
fn active_offsets_without_descriptor_tensors_are_rejected() {
    let mut offsets = LayerOffsets::zeroed();
    offsets.ffn_up = 2;
    assert!(plan_layer_tensor_window(&offsets, 0, 64, 2, &[]).is_err());
}

#[test]
fn fully_empty_layer_preserves_the_no_tensor_contract() {
    let offsets = LayerOffsets::zeroed();
    let (window, base) = plan_layer_tensor_window(&offsets, 17, 64, 2, &[]).unwrap();
    assert!(window.is_none());
    assert_eq!(base, 17);
}
