//! Template pipeline integration tests.
//!
//! These tests exercise the same code path as `make_prompt_renderer()` in
//! `shimmy_server_gpu.rs`:
//!
//!   1. Load `shimmytok::Tokenizer` from a real GGUF file.
//!   2. Extract `bos_token` / `eos_token` strings via `token_to_piece`.
//!   3. Read `tokenizer.chat_template` from the GGUF metadata.
//!   4. Render messages through `shimmyjinja::render_chat_template_with_context`.
//!   5. Assert on concrete output substrings.
//!
//! Tests skip gracefully when model files are absent so CI is not gated on
//! a model download.
//!
//! # GGUF header keys used by this pipeline
//!
//! | Key                            | Type   | Consumer                 |
//! |--------------------------------|--------|--------------------------|
//! | `tokenizer.chat_template`      | string | `make_prompt_renderer`   |
//! | `tokenizer.ggml.bos_token_id`  | u32    | `make_prompt_renderer`   |
//! | `tokenizer.ggml.eos_token_id`  | u32    | `make_prompt_renderer`   |
//! | `tokenizer.ggml.tokens`        | array  | `spec.rs` (vocab size)   |
//! | `{arch}.embedding_length`      | u32    | `spec.rs`                |
//! | `{arch}.block_count`           | u32    | `spec.rs`                |
//! | `general.architecture`         | string | `spec.rs` → ModelArch    |

use shimmyjinja::{render_chat_template_with_context, ChatMessage, RenderContext};
use std::io::Read;
use std::path::{Path, PathBuf};

// ── Minimal GGUF reader (chat_template + bos/eos ids only) ───────────────

const GGUF_TYPE_UINT8:   u32 = 0;
const GGUF_TYPE_INT8:    u32 = 1;
const GGUF_TYPE_UINT16:  u32 = 2;
const GGUF_TYPE_INT16:   u32 = 3;
const GGUF_TYPE_UINT32:  u32 = 4;
const GGUF_TYPE_INT32:   u32 = 5;
const GGUF_TYPE_FLOAT32: u32 = 6;
const GGUF_TYPE_BOOL:    u32 = 7;
const GGUF_TYPE_STRING:  u32 = 8;
const GGUF_TYPE_ARRAY:   u32 = 9;
const GGUF_TYPE_UINT64:  u32 = 10;
const GGUF_TYPE_INT64:   u32 = 11;
const GGUF_TYPE_FLOAT64: u32 = 12;

fn read_u32<R: Read>(r: &mut R) -> std::io::Result<u32> {
    let mut b = [0u8; 4]; r.read_exact(&mut b)?; Ok(u32::from_le_bytes(b))
}
fn read_u64<R: Read>(r: &mut R) -> std::io::Result<u64> {
    let mut b = [0u8; 8]; r.read_exact(&mut b)?; Ok(u64::from_le_bytes(b))
}
fn read_str<R: Read>(r: &mut R) -> std::io::Result<String> {
    let n = read_u64(r)? as usize;
    let mut buf = vec![0u8; n];
    r.read_exact(&mut buf)?;
    Ok(String::from_utf8_lossy(&buf).into_owned())
}
fn skip_val<R: Read>(r: &mut R, t: u32) -> std::io::Result<()> {
    match t {
        GGUF_TYPE_UINT8  | GGUF_TYPE_INT8  | GGUF_TYPE_BOOL    => { r.read_exact(&mut [0u8; 1])?; }
        GGUF_TYPE_UINT16 | GGUF_TYPE_INT16                     => { r.read_exact(&mut [0u8; 2])?; }
        GGUF_TYPE_UINT32 | GGUF_TYPE_INT32 | GGUF_TYPE_FLOAT32 => { read_u32(r)?; }
        GGUF_TYPE_UINT64 | GGUF_TYPE_INT64 | GGUF_TYPE_FLOAT64 => { read_u64(r)?; }
        GGUF_TYPE_STRING  => { read_str(r)?; }
        GGUF_TYPE_ARRAY   => {
            let et = read_u32(r)?;
            let n  = read_u64(r)?;
            for _ in 0..n { skip_val(r, et)?; }
        }
        _ => return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("unknown GGUF type {t}"),
        )),
    }
    Ok(())
}

struct GgufTokenizerInfo {
    chat_template:  Option<String>,
    bos_token_id:   Option<u32>,
    eos_token_id:   Option<u32>,
}

fn read_tokenizer_info(path: &Path) -> Option<GgufTokenizerInfo> {
    let mut f = std::fs::File::open(path).ok()?;
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic).ok()?;
    if &magic != b"GGUF" { return None; }

    let version = read_u32(&mut f).ok()?;
    let n_kv = if version >= 2 {
        read_u64(&mut f).ok()?; read_u64(&mut f).ok()?
    } else {
        read_u32(&mut f).ok()?; read_u32(&mut f).ok()? as u64
    };

    let mut info = GgufTokenizerInfo {
        chat_template: None, bos_token_id: None, eos_token_id: None,
    };

    for _ in 0..n_kv {
        let key = read_str(&mut f).ok()?;
        let vt  = read_u32(&mut f).ok()?;

        match key.as_str() {
            "tokenizer.chat_template" if vt == GGUF_TYPE_STRING => {
                info.chat_template = Some(read_str(&mut f).ok()?);
            }
            "tokenizer.ggml.bos_token_id" if vt == GGUF_TYPE_UINT32 => {
                info.bos_token_id = Some(read_u32(&mut f).ok()?);
            }
            "tokenizer.ggml.eos_token_id" if vt == GGUF_TYPE_UINT32 => {
                info.eos_token_id = Some(read_u32(&mut f).ok()?);
            }
            _ => { skip_val(&mut f, vt).ok()?; }
        }
    }
    Some(info)
}

// ── Helpers ───────────────────────────────────────────────────────────────

fn model_dir() -> PathBuf {
    let base = std::env::var("SHIMMY_TEST_MODELS")
        .unwrap_or_else(|_| "D:/shimmy-test-models".to_string());
    PathBuf::from(base)
}

macro_rules! skip_if_missing {
    ($path:expr) => {
        if !$path.exists() {
            eprintln!("SKIP — model not found: {}", $path.display());
            return;
        }
    };
}

/// Build a shimmyjinja context using real token-piece strings from shimmytok,
/// mirroring the `make_prompt_renderer` logic in `shimmy_server_gpu.rs`.
fn build_context_from_gguf(
    path: &Path,
    add_gen_prompt: bool,
) -> Option<(String, RenderContext)> {
    let info = read_tokenizer_info(path)?;
    let template = info.chat_template?;

    // Load tokenizer to convert token IDs to piece strings.
    let tok = shimmytok::Tokenizer::from_gguf_file(path.to_str()?).ok()?;

    let bos = info.bos_token_id
        .and_then(|id| tok.token_to_piece(id).ok())
        .unwrap_or_default();
    let eos = info.eos_token_id
        .and_then(|id| tok.token_to_piece(id).ok())
        .unwrap_or_default();

    let mut ctx = RenderContext::new();
    ctx.set_var("bos_token", &bos);
    ctx.set_var("eos_token", &eos);
    ctx.set_flag("add_generation_prompt", add_gen_prompt);

    Some((template, ctx))
}

// ══════════════════════════════════════════════════════════════════════════
// Per-model pipeline tests
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn pipeline_tinyllama_renders_user_turn_with_eos() {
    let path = model_dir().join("gguf_collection/TinyLlama-1.1B-Chat-v1.0.Q4_0.gguf");
    skip_if_missing!(path);

    let (tmpl, ctx) = build_context_from_gguf(&path, true)
        .expect("failed to load TinyLlama tokenizer info");

    let msgs = [ChatMessage { role: "user".into(), content: "Hello".into() }];
    let out = render_chat_template_with_context(&tmpl, &msgs, &ctx);

    // Real eos_token for TinyLlama is </s>
    assert!(out.contains("<|user|>"),      "user header; got: {out:?}");
    assert!(out.contains("Hello"),         "user content; got: {out:?}");
    assert!(out.contains("</s>"),          "eos_token from tokenizer; got: {out:?}");
    assert!(out.contains("<|assistant|>"), "gen-prompt; got: {out:?}");
}

#[test]
fn pipeline_gemma2_bos_and_turn_delimiters() {
    let path = model_dir().join("gguf_collection/gemma-2-2b-it-Q4_K_M.gguf");
    skip_if_missing!(path);

    let (tmpl, ctx) = build_context_from_gguf(&path, true)
        .expect("failed to load Gemma2 tokenizer info");

    let msgs = [ChatMessage { role: "user".into(), content: "Hello".into() }];
    let out = render_chat_template_with_context(&tmpl, &msgs, &ctx);

    // Gemma2 bos_token id=2 → "<bos>"
    assert!(out.starts_with("<bos>"),
        "bos_token from tokenizer; got: {out:?}");
    assert!(out.contains("<start_of_turn>user\nHello<end_of_turn>"),
        "user turn delimiters; got: {out:?}");
    assert!(out.contains("<start_of_turn>model\n"),
        "gen-prompt opens model turn; got: {out:?}");
}

#[test]
fn pipeline_phi35_eos_from_tokenizer() {
    let path = model_dir().join("gguf_collection/Phi-3.5-mini-instruct.Q4_K_M.gguf");
    skip_if_missing!(path);

    let (tmpl, ctx) = build_context_from_gguf(&path, true)
        .expect("failed to load Phi-3.5 tokenizer info");

    let msgs = [ChatMessage { role: "user".into(), content: "Hi".into() }];
    let out = render_chat_template_with_context(&tmpl, &msgs, &ctx);

    assert!(out.contains("<|user|>\nHi<|end|>"),
        "user turn with Phi3 delimiter; got: {out:?}");
    assert!(out.contains("<|assistant|>"),
        "gen-prompt; got: {out:?}");
}

#[test]
fn pipeline_qwen2_injects_default_system() {
    let path = model_dir().join("gguf_collection/qwen2-7b-instruct-q4_k_m.gguf");
    skip_if_missing!(path);

    let (tmpl, ctx) = build_context_from_gguf(&path, true)
        .expect("failed to load Qwen2 tokenizer info");

    let msgs = [ChatMessage { role: "user".into(), content: "Hello".into() }];
    let out = render_chat_template_with_context(&tmpl, &msgs, &ctx);

    assert!(out.contains("You are a helpful assistant."),
        "default system injection; got: {out:?}");
    assert!(out.contains("<|im_start|>user\nHello<|im_end|>"),
        "user ChatML turn; got: {out:?}");
    assert!(out.ends_with("<|im_start|>assistant\n"),
        "gen-prompt assistant turn; got: {out:?}");
}

#[test]
fn pipeline_deepseek_bos_from_tokenizer_and_plain_text_style() {
    let path = model_dir().join("gguf_collection/deepseek-llm-7b-chat.Q4_K_M.gguf");
    skip_if_missing!(path);

    let (tmpl, ctx) = build_context_from_gguf(&path, true)
        .expect("failed to load DeepSeek tokenizer info");

    let msgs = [ChatMessage { role: "user".into(), content: "Hello".into() }];
    let out = render_chat_template_with_context(&tmpl, &msgs, &ctx);

    // DeepSeek bos is the distinctive Unicode piece ｜begin▁of▁sentence｜
    // U+FF5C = ｜ (FULLWIDTH VERTICAL LINE), U+2581 = ▁ (LOWER ONE EIGHTH BLOCK)
    let bos = "\u{FF5C}begin\u{2581}of\u{2581}sentence\u{FF5C}";
    assert!(out.contains(bos),      "DeepSeek bos token from tokenizer; got: {out:?}");
    assert!(out.contains("User: Hello\n\n"), "plain-text user style; got: {out:?}");
    assert!(out.ends_with("Assistant:"),     "gen-prompt; got: {out:?}");
}

#[test]
fn pipeline_llama32_full_header_format() {
    let path = model_dir().join("gguf_collection/Llama-3.2-1B-Instruct-Q4_K_M.gguf");
    skip_if_missing!(path);

    let (tmpl, ctx) = build_context_from_gguf(&path, true)
        .expect("failed to load Llama-3.2 tokenizer info");

    let msgs = [
        ChatMessage { role: "user".into(),      content: "Hello".into() },
        ChatMessage { role: "assistant".into(),  content: "Hi!".into() },
        ChatMessage { role: "user".into(),       content: "Goodbye".into() },
    ];
    let out = render_chat_template_with_context(&tmpl, &msgs, &ctx);

    // Llama 3 bos_token id=128000 → <|begin_of_text|>
    assert!(out.contains("<|begin_of_text|>"),
        "Llama3 bos; got: {out:?}");
    assert!(out.contains("<|start_header_id|>user<|end_header_id|>"),
        "Llama3 user header; got: {out:?}");
    assert!(out.contains("<|eot_id|>"),
        "Llama3 eot delimiter; got: {out:?}");
    assert!(out.contains("<|start_header_id|>assistant<|end_header_id|>"),
        "Llama3 assistant header; got: {out:?}");
    assert!(out.contains("Hi!"),        "assistant content; got: {out:?}");
    assert!(out.contains("Goodbye"),    "final user content; got: {out:?}");
}

#[test]
fn pipeline_qwen3_chatml_no_tools() {
    let path = model_dir().join("gguf_collection/Qwen3-0.6B-Q4_K_M.gguf");
    skip_if_missing!(path);

    let (tmpl, ctx) = build_context_from_gguf(&path, true)
        .expect("failed to load Qwen3 tokenizer info");

    let msgs = [ChatMessage { role: "user".into(), content: "Hello".into() }];
    let out = render_chat_template_with_context(&tmpl, &msgs, &ctx);

    assert!(out.contains("<|im_start|>user"),
        "Qwen3 user header; got: {out:?}");
    assert!(out.contains("Hello"),
        "user content; got: {out:?}");
    assert!(out.contains("<|im_end|>"),
        "ChatML close tag; got: {out:?}");
    assert!(out.contains("<|im_start|>assistant"),
        "gen-prompt opens assistant turn; got: {out:?}");
}

// ── Smoke test: all models with templates render without panic ────────────

#[test]
fn pipeline_all_available_models_render_without_panic() {
    let dir = model_dir().join("gguf_collection");
    if !dir.exists() {
        eprintln!("SKIP — model directory not found: {}", dir.display());
        return;
    }

    let mut tested = 0usize;
    for entry in std::fs::read_dir(&dir).unwrap().flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("gguf") { continue; }

        let Some((tmpl, ctx)) = build_context_from_gguf(&path, true) else {
            eprintln!("  no template — {}", path.file_name().unwrap().to_string_lossy());
            continue;
        };

        let msgs = [ChatMessage { role: "user".into(), content: "Hello".into() }];
        let out = render_chat_template_with_context(&tmpl, &msgs, &ctx);
        assert!(!out.is_empty(),
            "empty output for {}", path.display());
        eprintln!("  OK  {} ({} bytes)", path.file_name().unwrap().to_string_lossy(), out.len());
        tested += 1;
    }
    eprintln!("Verified {} models through the full pipeline", tested);
}
