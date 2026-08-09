//! Integration tests for raw_gguf via the protocol crate.

use airframe_conformance_protocol::raw_gguf::{ModelIdentity, RawGgufReader};
use std::fs::File;
use std::io::{Seek, Write};
use tempfile::tempdir;

/// Create a minimal valid GGUF file for testing with tensor data.
fn create_minimal_gguf(dir: &std::path::Path) -> std::path::PathBuf {
    let path = dir.join("minimal.gguf");
    let mut file = File::create(&path).unwrap();

    // Magic: "GGUF"
    file.write_all(b"GGUF").unwrap();
    // Version: 3
    file.write_all(&3u32.to_le_bytes()).unwrap();
    // Tensor count: 1
    file.write_all(&1u64.to_le_bytes()).unwrap();
    // Metadata KV count: 4
    file.write_all(&4u64.to_le_bytes()).unwrap();

    // Metadata: general.architecture = "llama"
    write_string(&mut file, "general.architecture").unwrap();
    file.write_all(&8u32.to_le_bytes()).unwrap(); // STRING type
    write_string(&mut file, "llama").unwrap();

    // Metadata: general.alignment = 32
    write_string(&mut file, "general.alignment").unwrap();
    file.write_all(&4u32.to_le_bytes()).unwrap(); // UINT32 type
    file.write_all(&32u32.to_le_bytes()).unwrap();

    // Metadata: llama.block_count = 2
    write_string(&mut file, "llama.block_count").unwrap();
    file.write_all(&4u32.to_le_bytes()).unwrap(); // UINT32 type
    file.write_all(&2u32.to_le_bytes()).unwrap();

    // Metadata: llama.embedding_length = 512
    write_string(&mut file, "llama.embedding_length").unwrap();
    file.write_all(&4u32.to_le_bytes()).unwrap(); // UINT32 type
    file.write_all(&512u32.to_le_bytes()).unwrap();

    // Tensor: token_embd.weight
    // Name
    write_string(&mut file, "token_embd.weight").unwrap();
    // n_dims = 2
    file.write_all(&2u32.to_le_bytes()).unwrap();
    // dims: [100, 10]  -- small for testing
    file.write_all(&100u64.to_le_bytes()).unwrap();
    file.write_all(&10u64.to_le_bytes()).unwrap();
    // type: F32 (0)
    file.write_all(&0u32.to_le_bytes()).unwrap();
    // offset: 0 (relative to data_start)
    file.write_all(&0u64.to_le_bytes()).unwrap();

    // Align to 32 bytes
    let pos = file.stream_position().unwrap();
    let aligned = ((pos + 31) / 32) * 32;
    let padding = aligned - pos;
    file.write_all(&vec![0u8; padding as usize]).unwrap();

    // Write tensor data: 100 * 10 * 4 = 4000 bytes (F32)
    let tensor_data_size = 100u64 * 10 * 4;
    file.write_all(&vec![0u8; tensor_data_size as usize])
        .unwrap();

    file.flush().unwrap();
    path
}

/// Create a minimal valid GGUF file with multiple tensors.
fn create_multi_tensor_gguf(dir: &tempfile::TempDir) -> std::path::PathBuf {
    let path = dir.path().join("multi.gguf");
    let mut file = File::create(&path).unwrap();

    // Header
    file.write_all(b"GGUF").unwrap();
    file.write_all(&3u32.to_le_bytes()).unwrap();
    file.write_all(&2u64.to_le_bytes()).unwrap(); // 2 tensors
    file.write_all(&1u64.to_le_bytes()).unwrap(); // 1 metadata

    // Metadata: alignment
    write_string(&mut file, "general.alignment").unwrap();
    file.write_all(&4u32.to_le_bytes()).unwrap();
    file.write_all(&32u32.to_le_bytes()).unwrap();

    // Tensor 1: token_embd.weight (F16)
    write_string(&mut file, "token_embd.weight").unwrap();
    file.write_all(&2u32.to_le_bytes()).unwrap();
    file.write_all(&10u64.to_le_bytes()).unwrap();
    file.write_all(&10u64.to_le_bytes()).unwrap();
    file.write_all(&1u32.to_le_bytes()).unwrap(); // F16
    file.write_all(&0u64.to_le_bytes()).unwrap();

    // Tensor 2: output.weight (F32)
    write_string(&mut file, "output.weight").unwrap();
    file.write_all(&2u32.to_le_bytes()).unwrap();
    file.write_all(&10u64.to_le_bytes()).unwrap();
    file.write_all(&10u64.to_le_bytes()).unwrap();
    file.write_all(&0u32.to_le_bytes()).unwrap(); // F32
                                                  // offset: tensor1_size = 10*10*2 = 200 bytes
    file.write_all(&(10u64 * 10 * 2).to_le_bytes()).unwrap();

    // Align to 32 bytes
    let pos = file.stream_position().unwrap();
    let aligned = ((pos + 31) / 32) * 32;
    let padding = aligned - pos;
    file.write_all(&vec![0u8; padding as usize]).unwrap();

    // Tensor 1 data: 10 * 10 * 2 = 200 bytes (F16)
    let tensor1_size = 10u64 * 10 * 2;
    file.write_all(&vec![0u8; tensor1_size as usize]).unwrap();

    // Tensor 2 data: 10 * 10 * 4 = 400 bytes (F32)
    let tensor2_size = 10u64 * 10 * 4;
    file.write_all(&vec![0u8; tensor2_size as usize]).unwrap();

    file.flush().unwrap();
    path
}

fn write_string<W: Write>(w: &mut W, s: &str) -> std::io::Result<()> {
    w.write_all(&(s.len() as u64).to_le_bytes())?;
    w.write_all(s.as_bytes())?;
    Ok(())
}

/// Create a malformed GGUF (invalid magic).
fn create_invalid_magic_gguf(dir: &std::path::Path) -> std::path::PathBuf {
    let path = dir.join("bad_magic.gguf");
    let mut file = File::create(&path).unwrap();
    file.write_all(b"BAD ").unwrap(); // Wrong magic
    file.write_all(&3u32.to_le_bytes()).unwrap();
    file.write_all(&1u64.to_le_bytes()).unwrap();
    file.write_all(&0u64.to_le_bytes()).unwrap();
    path
}

/// Create a GGUF with unsupported version.
fn create_bad_version_gguf(dir: &std::path::Path) -> std::path::PathBuf {
    let path = dir.join("bad_version.gguf");
    let mut file = File::create(&path).unwrap();
    file.write_all(b"GGUF").unwrap();
    file.write_all(&99u32.to_le_bytes()).unwrap(); // Unsupported version
    file.write_all(&1u64.to_le_bytes()).unwrap();
    file.write_all(&0u64.to_le_bytes()).unwrap();
    path
}

/// Create a GGUF with invalid alignment.
fn create_bad_alignment_gguf(dir: &std::path::Path) -> std::path::PathBuf {
    let path = dir.join("bad_align.gguf");
    let mut file = File::create(&path).unwrap();
    file.write_all(b"GGUF").unwrap();
    file.write_all(&3u32.to_le_bytes()).unwrap();
    file.write_all(&1u64.to_le_bytes()).unwrap();
    file.write_all(&1u64.to_le_bytes()).unwrap();
    // Metadata: bad alignment (not multiple of 8)
    write_string(&mut file, "general.alignment").unwrap();
    file.write_all(&4u32.to_le_bytes()).unwrap();
    file.write_all(&33u32.to_le_bytes()).unwrap(); // 33 not multiple of 8
    file.write_all(&1u64.to_le_bytes()).unwrap(); // tensor count
    file.write_all(&0u64.to_le_bytes()).unwrap(); // metadata count
                                                  // Tensor
    write_string(&mut file, "t").unwrap();
    file.write_all(&1u32.to_le_bytes()).unwrap();
    file.write_all(&10u64.to_le_bytes()).unwrap();
    file.write_all(&0u32.to_le_bytes()).unwrap();
    file.write_all(&0u64.to_le_bytes()).unwrap();
    path
}

/// Create a GGUF with tensor offset out of bounds.
fn create_oob_offset_gguf(dir: &std::path::Path) -> std::path::PathBuf {
    let path = dir.join("oob_offset.gguf");
    let mut file = File::create(&path).unwrap();
    file.write_all(b"GGUF").unwrap();
    file.write_all(&3u32.to_le_bytes()).unwrap();
    file.write_all(&1u64.to_le_bytes()).unwrap();
    file.write_all(&1u64.to_le_bytes()).unwrap();
    // Metadata
    write_string(&mut file, "general.alignment").unwrap();
    file.write_all(&4u32.to_le_bytes()).unwrap();
    file.write_all(&32u32.to_le_bytes()).unwrap();
    // Tensor with offset past end of file
    write_string(&mut file, "bad_tensor").unwrap();
    file.write_all(&1u32.to_le_bytes()).unwrap();
    file.write_all(&10u64.to_le_bytes()).unwrap();
    file.write_all(&0u32.to_le_bytes()).unwrap();
    file.write_all(&999999u64.to_le_bytes()).unwrap(); // Offset way past end
    path
}

#[test]
fn raw_gguf_valid() {
    let dir = tempdir().unwrap();
    let path = create_minimal_gguf(dir.path());

    let reader = RawGgufReader::new();
    let identity = reader.read(&path).unwrap();

    assert_eq!(identity.version, 3);
    assert_eq!(identity.alignment, 32);
    assert_eq!(identity.tensors.len(), 1);
    assert_eq!(identity.tensors[0].name, "token_embd.weight");
    assert_eq!(identity.tensors[0].ggml_type, 0); // F32
    assert_eq!(identity.tensors[0].shape, vec![100, 10]);
    assert_eq!(identity.tensors[0].offset, identity.data_start_offset);
    assert!(identity.file_hash.len() == 64);
    assert!(identity.tensor_directory_hash.len() == 64);
}

#[test]
fn model_identity() {
    let dir = tempdir().unwrap();
    let path = create_minimal_gguf(dir.path());

    let reader = RawGgufReader::new();
    let id1 = reader.read(&path).unwrap();
    let id2 = reader.read(&path).unwrap();

    assert_eq!(id1.file_hash, id2.file_hash);
    assert_eq!(id1.tensor_directory_hash, id2.tensor_directory_hash);
    assert_eq!(id1, id2);
}

#[test]
fn raw_gguf_rejects_invalid_magic() {
    let dir = tempdir().unwrap();
    let path = create_invalid_magic_gguf(dir.path());

    let reader = RawGgufReader::new();
    let result = reader.read(&path);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.is_malformed());
}

#[test]
fn raw_gguf_rejects_bad_version() {
    let dir = tempdir().unwrap();
    let path = create_bad_version_gguf(dir.path());

    let reader = RawGgufReader::new();
    let result = reader.read(&path);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.is_malformed());
}

#[test]
fn raw_gguf_rejects_bad_alignment() {
    let dir = tempdir().unwrap();
    let path = create_bad_alignment_gguf(dir.path());

    let reader = RawGgufReader::new();
    let result = reader.read(&path);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.is_malformed());
}

#[test]
fn raw_gguf_rejects_oob_offset() {
    let dir = tempdir().unwrap();
    let path = create_oob_offset_gguf(dir.path());

    let reader = RawGgufReader::new();
    let result = reader.read(&path);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.is_malformed());
}
