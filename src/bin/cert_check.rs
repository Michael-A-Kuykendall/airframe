//! cert_check — FSE single-pass Rust source certification.
//!
//! Architecture: all selector patterns compiled into one Aho-Corasick DFA at
//! startup. The file is iterated once; every match is broadcast to all
//! registered criterion handlers simultaneously.
//!
//!   ∂runtime/∂criteria ≈ 0  (for shared selectors)
//!
//! This is the coin-sorter inversion: one pass, everything lands where it belongs.

use aho_corasick::AhoCorasick;
use clap::Parser;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

// ── CLI ───────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "cert_check", about = "FSE single-pass cert checker for Rust source")]
struct Cli {
    /// Source file(s) to check
    #[arg(required_unless_present = "list")]
    files: Vec<PathBuf>,

    /// Seal the file in CERT.toml if certifiable
    #[arg(long)]
    seal: bool,

    /// Add a waiver: CID:reason  e.g. --waive B6:debug bridge code
    #[arg(long, value_name = "CID:REASON")]
    waive: Option<String>,

    /// List all criteria and exit
    #[arg(long)]
    list: bool,
}

// ── FSE Selector Pattern Table ────────────────────────────────────────────────
//
// Each pattern is a selector. Multiple criteria can register on the same
// pattern — the dispatch match statement is the broadcast table.

const PAT: &[&str] = &[
    ".clone()",          //  0  B1  clone_pressure
    ".unwrap()",         //  1  B3  unwrap_density  |  C4  gpu_unwrap
    "TODO",              //  2  B2  todo_desert
    "FIXME",             //  3  B2  todo_desert
    "println!",          //  4  B6  debug_prints
    "eprintln!",         //  5  B6  debug_prints
    "dbg!(",             //  6  B6  debug_prints
    "#[allow(",          //  7  B4  lint_suppress  |  C6  dead_code_bare
    "dead_code",         //  8  C6  (context — used alongside PAT[7])
    "todo!()",           //  9  B8  prod_panics
    "unimplemented!()",  // 10  B8  prod_panics
    "panic!(",           // 11  B8  prod_panics
    "pub fn ",           // 12  B5  pub_pressure
    "pub struct ",       // 13  B5  pub_pressure
    "pub enum ",         // 14  B5  pub_pressure
    "pub trait ",        // 15  B5  pub_pressure
    "pub type ",         // 16  B5  pub_pressure
    "pub const ",        // 17  B5  pub_pressure
    "create_buffer(",    // 18  C5  buffer_labels
    "device.create_",    // 19  C4  GPU call-site context
    "queue.write_",      // 20  C4  GPU call-site context
    "queue.submit(",     // 21  C4  GPU call-site context
    "device.poll(",      // 22  C4  GPU call-site context
];

const P_CLONE: usize = 0;
const P_UNWRAP: usize = 1;
const P_TODO: usize = 2;
const P_FIXME: usize = 3;
const P_PRINTLN: usize = 4;
const P_EPRINTLN: usize = 5;
const P_DBG: usize = 6;
const P_ALLOW: usize = 7;
const P_DEAD_CODE: usize = 8;
const P_TODO_MACRO: usize = 9;
const P_UNIMPL: usize = 10;
const P_PANIC: usize = 11;
const P_PUB_FN: usize = 12;
// 13-17 are the other pub-item patterns; all handled the same way
const P_CREATE_BUF: usize = 18;
const P_DEVICE_CREATE: usize = 19;
// 20-22 are the other GPU context patterns

// ── Thresholds ────────────────────────────────────────────────────────────────

const LINE_COUNT_MAX: usize = 600;
const CLONE_PER_100_MAX: f64 = 6.0;
const UNWRAP_PER_1K_MAX: f64 = 15.0;
const PUB_PRESSURE_MAX: f64 = 0.70;
const MIN_FN_COUNT: usize = 5;

// ── Scan state ────────────────────────────────────────────────────────────────

#[derive(Default)]
struct ScanState {
    total_lines: usize,
    prod_lines: usize,

    // Criterion counters
    clone_count: usize,
    unwrap_prod: usize,
    fn_count: usize,        // fn keyword occurrences — used for C11 threshold
    pub_count: usize,       // pub-qualified declarations — B5 numerator
    total_decl_count: usize, // all declarations (pub+private) — B5 denominator

    // Flags
    found_todo: bool,

    // Issue collections  (line_no, snippet)
    gpu_unwrap_lines: Vec<(usize, String)>,
    debug_print_lines: Vec<(usize, String)>,
    allow_no_comment: Vec<(usize, String)>,
    dead_code_bare: Vec<(usize, String)>,
    vague_expect_lines: Vec<(usize, String)>,
    prod_panic_lines: Vec<(usize, String)>,
    placeholder_lines: Vec<(usize, String)>,
    create_buf_lines: Vec<usize>,       // raw positions for C5 post-process
    create_buf_unlabeled: Vec<usize>,   // filled after scan
    commented_block_issues: Vec<usize>,

    // Per-line context (updated at end of each iteration)
    prev_is_comment: bool,
    consec_commented_code: usize,

    // Test-block tracking (approximate; brace-counted)
    in_test: bool,
    has_test_module: bool,
    brace_depth: i32,
    test_entry_depth: i32,

    // GPU call-site flag for current line (set before AC loop, used inside it)
    gpu_call_on_line: bool,
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Heuristic: does a comment body look like commented-out Rust?
fn looks_like_code(comment_line: &str) -> bool {
    let inner = comment_line.trim_start_matches('/').trim();
    inner.contains('(')
        || inner.contains('{')
        || inner.contains(';')
        || inner.contains("let ")
        || inner.contains("return ")
        || inner.contains("for ")
        || inner.contains("if ")
        || inner.contains("->")
        || inner.contains("::")
}

const VAGUE_SUFFIXES: &[&str] = &[
    "\"error\"", "\"failed\"", "\"should", "\"unwrap\"",
    "\"none\"",  "\"some\"",   "\"ok\"",   "\"must\"",
    "\"todo\"",  "\"fixme\"",
];

fn is_vague_expect(line: &str) -> bool {
    let lower = line.to_lowercase();
    if !lower.contains(".expect(\"") {
        return false;
    }
    VAGUE_SUFFIXES.iter().any(|s| lower.contains(s))
}

const PLACEHOLDER_NAMES: &[&str] = &[
    "data", "buf", "tmp", "res", "result", "output",
    "handler", "value", "info", "ctx", "manager", "helper",
    "service", "util", "obj", "item",
];

/// Compute byte ranges of double-quoted string literals on a single line.
/// Used to exclude pattern matches that fall inside string contents.
/// Handles `\"` escapes. Does not handle raw strings or multi-line strings.
fn string_literal_spans(line: &[u8]) -> Vec<std::ops::Range<usize>> {
    let mut spans = Vec::new();
    let mut i = 0;
    while i < line.len() {
        if line[i] == b'"' {
            let start = i;
            i += 1;
            while i < line.len() {
                match line[i] {
                    b'\\' => i += 2,  // skip escaped char — won't overrun; clamped below
                    b'"' => {
                        i += 1;
                        spans.push(start..i);
                        break;
                    }
                    _ => i += 1,
                }
                i = i.min(line.len());
            }
        } else {
            i += 1;
        }
    }
    spans
}

/// True if a function-definition line has a placeholder name as a parameter.
fn has_placeholder_param(line: &str) -> bool {
    if !line.contains("fn ") || !line.contains('(') {
        return false;
    }
    for name in PLACEHOLDER_NAMES {
        // Match `name:` preceded by ` `, `(`, or `,`
        for sep in &[" ", "(", ","] {
            let needle = format!("{}{}", sep, name);
            if let Some(pos) = line.find(&needle) {
                let after = &line[pos + sep.len() + name.len()..];
                if after.starts_with(':') {
                    return true;
                }
            }
        }
    }
    false
}

// ── Main FSE scan ─────────────────────────────────────────────────────────────

fn scan(content: &str, ac: &AhoCorasick) -> ScanState {
    let mut st = ScanState::default();
    let lines: Vec<&str> = content.lines().collect();

    for (idx, &line) in lines.iter().enumerate() {
        let lineno = idx + 1;
        let tr = line.trim();
        let is_comment = tr.starts_with("//") && !tr.starts_with("///");
        let is_doc = tr.starts_with("///");

        // ── Test-block tracking (brace depth) ──────────────────────────────
        if tr.contains("#[cfg(test)]") && !is_comment {
            st.in_test = true;
            st.has_test_module = true;
            st.test_entry_depth = st.brace_depth;
        }
        for ch in line.chars() {
            match ch {
                '{' => st.brace_depth += 1,
                '}' => {
                    st.brace_depth -= 1;
                    if st.in_test && st.brace_depth <= st.test_entry_depth {
                        st.in_test = false;
                    }
                }
                _ => {}
            }
        }

        st.total_lines += 1;
        let skip_prod = st.in_test;
        if !skip_prod {
            st.prod_lines += 1;
        }

        // ── C7: consecutive commented-out-code lines ────────────────────────
        if is_comment && looks_like_code(tr) {
            st.consec_commented_code += 1;
            if st.consec_commented_code >= 3 {
                // Report the start of the block (3 lines up)
                let block_start = lineno.saturating_sub(2);
                if st.commented_block_issues.last().copied().unwrap_or(0) != block_start {
                    st.commented_block_issues.push(block_start);
                }
            }
        } else if !is_comment {
            st.consec_commented_code = 0;
        }

        // ── fn_count + total_decl_count ─────────────────────────────────────
        // fn_count: used by C11 test-coverage threshold.
        // total_decl_count: all declarable items (fn+struct+enum+trait+type+const,
        //   pub and private) — correct denominator for B5 pub-pressure ratio.
        if !is_comment {
            st.fn_count += line.matches("fn ").count();
            for kw in &["fn ", "struct ", "enum ", "trait ", "type ", "const "] {
                st.total_decl_count += line.matches(kw).count();
            }
        }

        // ── B7: vague expects ───────────────────────────────────────────────
        if !skip_prod && !is_comment && is_vague_expect(tr) {
            st.vague_expect_lines.push((lineno, tr.to_string()));
        }

        // ── B9: placeholder names in signatures ────────────────────────────
        if !is_comment && has_placeholder_param(tr) {
            st.placeholder_lines.push((lineno, tr.to_string()));
        }

        // ── GPU call-site flag (set before AC loop so P_UNWRAP can read it) ─
        st.gpu_call_on_line = !is_comment
            && (tr.contains("device.create_")
                || tr.contains("queue.write_")
                || tr.contains("queue.submit(")
                || tr.contains("device.poll("));

        // ── FSE dispatch: single AC scan → broadcast to all criteria ────────
        let mut saw_allow = false;
        let mut saw_dead_code = false;

        // Pre-compute string literal spans so the dispatch loop can exclude
        // matches that fall inside string contents (e.g. PAT array entries
        // in cert_check.rs itself, or "use println! to..." error strings).
        let str_spans = if !is_comment {
            string_literal_spans(line.as_bytes())
        } else {
            vec![]
        };

        for mat in ac.find_iter(line.as_bytes()) {
            // Skip matches whose start position is inside a string literal.
            if str_spans.iter().any(|r| r.contains(&mat.start())) {
                continue;
            }
            match mat.pattern().as_usize() {
                P_CLONE => {
                    if !is_comment {
                        st.clone_count += 1;
                    }
                }
                P_UNWRAP => {
                    if !is_comment && !skip_prod {
                        st.unwrap_prod += 1;            // B3
                        if st.gpu_call_on_line {
                            st.gpu_unwrap_lines.push((lineno, tr.to_string())); // C4
                        }
                    }
                }
                P_TODO | P_FIXME => {
                    st.found_todo = true;               // B2
                }
                P_PRINTLN | P_EPRINTLN | P_DBG => {
                    if !is_comment && !is_doc && !skip_prod {
                        st.debug_print_lines.push((lineno, tr.to_string())); // B6
                    }
                }
                P_ALLOW => {
                    if !is_comment {
                        saw_allow = true;               // B4, C6 — resolved after loop
                    }
                }
                P_DEAD_CODE => {
                    saw_dead_code = true;               // C6 context
                }
                P_TODO_MACRO | P_UNIMPL => {
                    if !is_comment && !skip_prod {
                        st.prod_panic_lines.push((lineno, tr.to_string())); // B8
                    }
                }
                P_PANIC => {
                    if !is_comment && !skip_prod && !st.prev_is_comment {
                        st.prod_panic_lines.push((lineno, tr.to_string())); // B8
                    }
                }
                P_PUB_FN..=17 => {
                    // Covers pub fn / pub struct / pub enum / pub trait / pub type / pub const
                    if !is_comment {
                        st.pub_count += 1;              // B5
                    }
                }
                P_CREATE_BUF => {
                    if !is_comment && !skip_prod {
                        st.create_buf_lines.push(lineno); // C5 — post-processed below
                    }
                }
                P_DEVICE_CREATE..=22 => {
                    // GPU context patterns — the gpu_call_on_line flag is already set above;
                    // these matches have no additional action in the dispatch table.
                }
                _ => {}
            }
        }

        // ── B4 + C6: post-AC resolution (requires saw_allow + saw_dead_code) ─
        if saw_allow {
            if !st.prev_is_comment {
                st.allow_no_comment.push((lineno, tr.to_string())); // B4
            }
            if saw_dead_code && !st.prev_is_comment {
                st.dead_code_bare.push((lineno, tr.to_string()));   // C6
            }
        }

        // ── Context update ──────────────────────────────────────────────────
        st.prev_is_comment = is_comment && !is_doc;
    }

    // ── C5 post-process: did each create_buffer() call include a label? ──────
    for &buf_lineno in &st.create_buf_lines {
        let start = buf_lineno.saturating_sub(1);
        let end = (buf_lineno + 9).min(lines.len());
        // Collect text from start of call until the first `;` or closing `)`
        let segment = lines[start..end].join("\n");
        let up_to_semi = segment.split(';').next().unwrap_or(&segment);
        if !up_to_semi.contains("label:") {
            st.create_buf_unlabeled.push(buf_lineno);
        }
    }

    st
}

// ── Criterion evaluation ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
enum Status {
    Pass,
    Warn,
    Fail,
    Skip,
    Waived,
}

impl Status {
    fn symbol(&self) -> &str {
        match self {
            Status::Pass => "PASS",
            Status::Warn => "WARN",
            Status::Fail => "FAIL",
            Status::Skip => "SKIP",
            Status::Waived => "WAIV",
        }
    }
    fn is_blocking(&self) -> bool {
        matches!(self, Status::Fail)
    }
}

struct CriterionResult {
    id: &'static str,
    status: Status,
    detail: String,
}

fn evaluate(st: &ScanState, waivers: &HashMap<String, String>) -> Vec<CriterionResult> {
    let mut results = Vec::new();

    macro_rules! criterion {
        ($id:expr, $status:expr, $detail:expr) => {{
            let status = if waivers.contains_key($id) {
                Status::Waived
            } else {
                $status
            };
            let detail = if let Some(reason) = waivers.get($id) {
                format!("{} [waiver: {}]", $detail, reason)
            } else {
                $detail
            };
            results.push(CriterionResult { id: $id, status, detail });
        }};
    }

    // C3: line_count — soft cap 600 (waiveable), hard cap 2000 (non-waiveable)
    const HARD_CAP_LINES: usize = 2000;
    {
        let (status, detail) = if st.total_lines > HARD_CAP_LINES {
            // Hard cap — never waiveable regardless of CERT.toml entry
            (Status::Fail,
             format!("{} lines exceeds hard cap {} — architectural split required (waiver not accepted)",
                st.total_lines, HARD_CAP_LINES))
        } else if st.total_lines > LINE_COUNT_MAX {
            if let Some(reason) = waivers.get("C3") {
                (Status::Waived, format!("{} lines (max {}) [waiver: {}]", st.total_lines, LINE_COUNT_MAX, reason))
            } else {
                (Status::Fail, format!("{} lines (max {})", st.total_lines, LINE_COUNT_MAX))
            }
        } else {
            (Status::Pass, format!("{} lines (max {})", st.total_lines, LINE_COUNT_MAX))
        };
        results.push(CriterionResult { id: "C3", status, detail });
    }

    // C4: gpu_unwrap
    criterion!(
        "C4",
        if st.gpu_unwrap_lines.is_empty() { Status::Pass } else { Status::Fail },
        if st.gpu_unwrap_lines.is_empty() {
            "no .unwrap() on GPU call sites".to_string()
        } else {
            format!("{} unwrap(s) on GPU call sites: lines {:?}", st.gpu_unwrap_lines.len(),
                st.gpu_unwrap_lines.iter().map(|(l, _)| l).collect::<Vec<_>>())
        }
    );

    // C5: buffer_labels
    criterion!(
        "C5",
        if st.create_buf_unlabeled.is_empty() { Status::Pass } else { Status::Fail },
        if st.create_buf_unlabeled.is_empty() {
            format!("all {} create_buffer() calls labeled", st.create_buf_lines.len())
        } else {
            format!("unlabeled create_buffer at lines {:?}", st.create_buf_unlabeled)
        }
    );

    // C6: dead_code_bare
    criterion!(
        "C6",
        if st.dead_code_bare.is_empty() { Status::Pass } else { Status::Fail },
        if st.dead_code_bare.is_empty() {
            "no bare #[allow(dead_code)]".to_string()
        } else {
            format!("{} bare dead_code suppress(es) at lines {:?}",
                st.dead_code_bare.len(),
                st.dead_code_bare.iter().map(|(l, _)| l).collect::<Vec<_>>())
        }
    );

    // C7: commented_blocks
    criterion!(
        "C7",
        if st.commented_block_issues.is_empty() { Status::Pass } else { Status::Fail },
        if st.commented_block_issues.is_empty() {
            "no commented-out code blocks".to_string()
        } else {
            format!("commented-out code blocks starting at lines {:?}", st.commented_block_issues)
        }
    );

    // B1: clone_pressure
    let clone_rate = if st.total_lines > 0 {
        st.clone_count as f64 / st.total_lines as f64 * 100.0
    } else {
        0.0
    };
    criterion!(
        "B1",
        if clone_rate <= CLONE_PER_100_MAX { Status::Pass } else { Status::Fail },
        format!("{} .clone()s = {:.1}/100 lines (max {:.0})", st.clone_count, clone_rate, CLONE_PER_100_MAX)
    );

    // B2: todo_desert — FAIL for large files (> 1000 lines) with no TODO/FIXME; WARN otherwise
    const TODO_DESERT_LARGE: usize = 1000;
    let b2_status = if st.found_todo {
        Status::Pass
    } else if st.total_lines > TODO_DESERT_LARGE {
        Status::Fail  // large file + no acknowledged debt = implausible; P1 AI signal
    } else {
        Status::Warn
    };
    criterion!(
        "B2",
        b2_status,
        if st.found_todo {
            "TODO/FIXME present".to_string()
        } else if st.total_lines > TODO_DESERT_LARGE {
            format!("no TODO/FIXME in {}-line file — AI desert FAIL (P1 signal); add at minimum one TODO",
                st.total_lines)
        } else {
            "no TODO/FIXME — possible AI-generated desert (P1 signal)".to_string()
        }
    );

    // B3: unwrap_density
    let unwrap_rate = if st.prod_lines > 0 {
        st.unwrap_prod as f64 / st.prod_lines as f64 * 1000.0
    } else {
        0.0
    };
    let b3_status = if unwrap_rate <= UNWRAP_PER_1K_MAX { Status::Pass }
        else if unwrap_rate <= UNWRAP_PER_1K_MAX * 1.5 { Status::Warn }
        else { Status::Fail };
    criterion!(
        "B3",
        b3_status,
        format!("{} unwrap(s) = {:.1}/1k prod lines (max {})", st.unwrap_prod, unwrap_rate, UNWRAP_PER_1K_MAX)
    );

    // B4: lint_suppress
    criterion!(
        "B4",
        if st.allow_no_comment.is_empty() { Status::Pass } else { Status::Fail },
        if st.allow_no_comment.is_empty() {
            "all #[allow(..)] have justification comments".to_string()
        } else {
            format!("{} unjustified #[allow(..)] at lines {:?}",
                st.allow_no_comment.len(),
                st.allow_no_comment.iter().map(|(l, _)| l).collect::<Vec<_>>())
        }
    );

    // B5: pub_pressure
    // Ratio = pub declarations / ALL declarations (pub+private).
    // Denominator is total_decl_count (fn+struct+enum+trait+type+const), NOT fn_count.
    // Using fn_count alone was a bug: pub structs/enums inflated the numerator
    // against a fn-only denominator, producing ratios > 100% on type-heavy files.
    let b5 = if st.total_decl_count < MIN_FN_COUNT {
        (Status::Skip, format!("too few declarations ({} < {})", st.total_decl_count, MIN_FN_COUNT))
    } else {
        let ratio = st.pub_count as f64 / st.total_decl_count as f64;
        if ratio <= PUB_PRESSURE_MAX {
            (Status::Pass, format!("{}/{} = {:.0}%", st.pub_count, st.total_decl_count, ratio * 100.0))
        } else {
            (Status::Fail, format!("{}/{} = {:.0}% > {:.0}% (C3 AI signal)", st.pub_count, st.total_decl_count, ratio * 100.0, PUB_PRESSURE_MAX * 100.0))
        }
    };
    criterion!("B5", b5.0, b5.1);

    // B6: debug_prints
    criterion!(
        "B6",
        if st.debug_print_lines.is_empty() { Status::Pass } else { Status::Fail },
        if st.debug_print_lines.is_empty() {
            "no raw println!/eprintln!/dbg!".to_string()
        } else {
            format!("{} debug print(s) at lines {:?} (A6 perfect discriminator)",
                st.debug_print_lines.len(),
                st.debug_print_lines.iter().map(|(l, _)| l).collect::<Vec<_>>())
        }
    );

    // B7: vague_expects
    criterion!(
        "B7",
        if st.vague_expect_lines.is_empty() { Status::Pass } else { Status::Fail },
        if st.vague_expect_lines.is_empty() {
            "no vague .expect() messages".to_string()
        } else {
            format!("{} vague expect(s) at lines {:?}",
                st.vague_expect_lines.len(),
                st.vague_expect_lines.iter().map(|(l, _)| l).collect::<Vec<_>>())
        }
    );

    // B8: prod_panics
    criterion!(
        "B8",
        if st.prod_panic_lines.is_empty() { Status::Pass } else { Status::Fail },
        if st.prod_panic_lines.is_empty() {
            "no todo!()/unimplemented!()/uncommented panic!()".to_string()
        } else {
            format!("{} prod panic(s) at lines {:?}",
                st.prod_panic_lines.len(),
                st.prod_panic_lines.iter().map(|(l, _)| l).collect::<Vec<_>>())
        }
    );

    // B9: placeholder_names
    criterion!(
        "B9",
        if st.placeholder_lines.is_empty() { Status::Pass } else { Status::Fail },
        if st.placeholder_lines.is_empty() {
            "no placeholder param names in signatures".to_string()
        } else {
            format!("{} placeholder param(s) at lines {:?}",
                st.placeholder_lines.len(),
                st.placeholder_lines.iter().map(|(l, _)| l).collect::<Vec<_>>())
        }
    );

    // C11: test_coverage — FAIL if > 5 functions and no #[cfg(test)] module present
    const MIN_FNS_FOR_TEST: usize = 5;
    let c11_status = if st.has_test_module {
        Status::Pass
    } else if st.fn_count > MIN_FNS_FOR_TEST {
        Status::Fail
    } else {
        Status::Skip
    };
    criterion!(
        "C11",
        c11_status,
        if st.has_test_module {
            "has #[cfg(test)] module".to_string()
        } else if st.fn_count > MIN_FNS_FOR_TEST {
            format!("{} functions, no #[cfg(test)] module — pure-function code must have unit tests",
                st.fn_count)
        } else {
            format!("{} functions (skip threshold {})", st.fn_count, MIN_FNS_FOR_TEST)
        }
    );

    results
}

// ── CERT.toml I/O ─────────────────────────────────────────────────────────────

fn cert_toml_path(source_file: &Path) -> PathBuf {
    // Walk up to find the workspace root (contains Cargo.toml at top level)
    let mut dir = source_file.canonicalize().unwrap_or_else(|_| source_file.to_path_buf());
    dir.pop();
    loop {
        if dir.join("Cargo.toml").exists() && dir.join("src").exists() {
            return dir.join("CERT.toml");
        }
        if !dir.pop() {
            break;
        }
    }
    PathBuf::from("CERT.toml")
}

fn workspace_relative(source_file: &Path, cert_path: &Path) -> String {
    let abs = source_file.canonicalize().unwrap_or_else(|_| source_file.to_path_buf());
    let root = cert_path.parent().unwrap_or(Path::new("."));
    abs.strip_prefix(root)
        .map(|p| p.to_string_lossy().replace('\\', "/"))
        .unwrap_or_else(|_| source_file.to_string_lossy().replace('\\', "/"))
}

/// Read waivers for a given file path from CERT.toml.
fn read_waivers(cert_text: &str, rel_path: &str) -> HashMap<String, String> {
    let section = format!("[files.\"{}\".waivers]", rel_path);
    let mut waivers = HashMap::new();
    let mut in_section = false;
    for line in cert_text.lines() {
        let t = line.trim();
        if t == section {
            in_section = true;
            continue;
        }
        if in_section {
            if t.starts_with('[') {
                break;
            }
            // Parse  KEY = "VALUE"
            if let Some((k, v)) = t.split_once('=') {
                let k = k.trim().to_string();
                let v = v.trim().trim_matches('"').to_string();
                waivers.insert(k, v);
            }
        }
    }
    waivers
}

/// Write a waiver entry to CERT.toml.
fn write_waiver(cert_path: &Path, rel_path: &str, cid: &str, reason: &str) -> std::io::Result<()> {
    let existing = if cert_path.exists() {
        std::fs::read_to_string(cert_path)?
    } else {
        String::new()
    };

    let waiver_section = format!("[files.\"{}\".waivers]", rel_path);

    // Check if the section already exists
    if existing.contains(&waiver_section) {
        // Insert the new key after the section header
        let mut out = String::new();
        let mut in_section = false;
        let mut inserted = false;
        for line in existing.lines() {
            if line.trim() == waiver_section {
                in_section = true;
                out.push_str(line);
                out.push('\n');
                continue;
            }
            if in_section && !inserted {
                // Check if key already exists — overwrite it
                if line.trim().starts_with(&format!("{} =", cid)) {
                    out.push_str(&format!("{} = \"{}\"\n", cid, reason));
                    inserted = true;
                    continue;
                }
                // Insert before next section
                if line.trim().starts_with('[') {
                    out.push_str(&format!("{} = \"{}\"\n", cid, reason));
                    inserted = true;
                    in_section = false;
                }
            }
            out.push_str(line);
            out.push('\n');
        }
        if !inserted {
            // Was at end of file
            out.push_str(&format!("{} = \"{}\"\n", cid, reason));
        }
        std::fs::write(cert_path, out)
    } else {
        // Append a new section
        let mut out = existing;
        if !out.is_empty() && !out.ends_with('\n') {
            out.push('\n');
        }
        out.push_str(&format!("\n[files.\"{}\".waivers]\n", rel_path));
        out.push_str(&format!("{} = \"{}\"\n", cid, reason));
        std::fs::write(cert_path, out)
    }
}

/// Write a seal entry to CERT.toml.
fn write_seal(cert_path: &Path, rel_path: &str, sha256: &str, date: &str) -> std::io::Result<()> {
    let existing = if cert_path.exists() {
        std::fs::read_to_string(cert_path)?
    } else {
        String::new()
    };

    let file_section = format!("[files.\"{}\"]", rel_path);
    let new_seal = format!(
        "[files.\"{}\"]\nsealed_sha256 = \"{}\"\nsealed_date = \"{}\"\n",
        rel_path, sha256, date
    );

    if existing.contains(&file_section) {
        // Replace existing block (up to next top-level section or EOF)
        let mut out = String::new();
        let mut in_section = false;
        let mut replaced = false;
        for line in existing.lines() {
            if line.trim() == file_section {
                in_section = true;
                if !replaced {
                    out.push_str(&new_seal);
                    replaced = true;
                }
                continue;
            }
            if in_section {
                if line.trim().starts_with('[') {
                    in_section = false;
                    out.push_str(line);
                    out.push('\n');
                }
                // Skip old seal lines
                continue;
            }
            out.push_str(line);
            out.push('\n');
        }
        std::fs::write(cert_path, out)
    } else {
        let mut out = existing;
        if !out.is_empty() && !out.ends_with('\n') {
            out.push('\n');
        }
        out.push('\n');
        out.push_str(&new_seal);
        std::fs::write(cert_path, out)
    }
}

fn sha256_hex(path: &Path) -> std::io::Result<String> {
    let bytes = std::fs::read(path)?;
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

// ── Output ────────────────────────────────────────────────────────────────────

fn print_results(path: &Path, results: &[CriterionResult], st: &ScanState) {
    println!("\n{}", path.display());
    println!("{}", "─".repeat(60));
    for r in results {
        let line = format!("  [{:4}] {:<4}  {}", r.status.symbol(), r.id, r.detail);
        println!("{}", line);
    }
    println!("{}", "─".repeat(60));

    let blocking: Vec<_> = results.iter().filter(|r| r.status.is_blocking()).collect();
    let certifiable = blocking.is_empty();

    let warn_count = results.iter().filter(|r| r.status == Status::Warn).count();
    let pass_count = results.iter().filter(|r| r.status == Status::Pass).count();
    let waived_count = results.iter().filter(|r| r.status == Status::Waived).count();

    println!("  {} lines | {} pass | {} warn | {} fail | {} waived",
        st.total_lines, pass_count, warn_count, blocking.len(), waived_count);

    if certifiable {
        if waived_count > 0 {
            println!("  Certifiable: YES  ({} waiver(s) active — review before merge)", waived_count);
        } else {
            println!("  Certifiable: YES");
        }
    } else {
        println!("  Certifiable: NO  ({} blocking failure(s))", blocking.len());
        for r in &blocking {
            println!("    ✗ {}: {}", r.id, r.detail);
        }
    }
}

fn print_criteria_list() {
    println!("Auto criteria (checked by cert_check):");
    let auto_criteria = [
        ("C3", "line_count",       "file <= 600 lines"),
        ("C4", "gpu_unwrap",       "no .unwrap() on GPU (device/queue) call sites"),
        ("C5", "buffer_labels",    "all create_buffer() calls have label:"),
        ("C6", "dead_code_bare",   "no bare #[allow(dead_code)] without preceding comment"),
        ("C7", "commented_blocks", "no 3+ consecutive commented-out code lines"),
        ("B1", "clone_pressure",   "<= 6 .clone() / 100 lines  (P6: 24x AI signal)"),
        ("B2", "todo_desert",      "file has at least one TODO/FIXME  (P1: 5x AI signal)"),
        ("B3", "unwrap_density",   "<= 15 .unwrap() / 1k prod lines  (N5)"),
        ("B4", "lint_suppress",    "all #[allow(..)] have justification comments  (P7)"),
        ("B5", "pub_pressure",     "pub items <= 70% of fns, min 5 fns  (C3: 12.6x AI signal)"),
        ("B6", "debug_prints",     "no raw println!/eprintln!/dbg!  (A6: perfect discriminator)"),
        ("B7", "vague_expects",    "no .expect(\"error\"/\"failed\"/etc)  (A4)"),
        ("B8", "prod_panics",      "no todo!()/unimplemented!(); panic! needs preceding comment  (A3)"),
        ("B9", "placeholder_names","no data/buf/tmp/res/etc in function signatures  (D11)"),
        ("C11","test_coverage",    "has #[cfg(test)] module when fn_count > 5"),
    ];
    for (id, name, desc) in &auto_criteria {
        println!("  {:<4} {:<20} {}", id, name, desc);
    }
    println!();
    println!("Manual criteria (record in CERT.toml by hand):");
    println!("  C1   clippy           cargo clippy -- -D warnings");
    println!("  C2   rustfmt          cargo fmt --check");
    println!("  C8   no_todos         no unresolved TODO/FIXMEs before seal");
    println!("  C9   quant_verify     quant_verify passes for this model");
    println!("  C10  smoke_test       end-to-end smoke test passes");
}

// ── Entry point ───────────────────────────────────────────────────────────────

fn main() {
    let cli = Cli::parse();

    if cli.list {
        print_criteria_list();
        return;
    }

    // Compile all selector patterns into one DFA — the FSE compilation step.
    // Cost is paid once; criteria count does not affect scan time.
    let ac = AhoCorasick::new(PAT).expect("cert_check: pattern compile failed");

    let mut any_uncertifiable = false;

    for file_path in &cli.files {
        let content = match std::fs::read_to_string(file_path) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("cert_check: cannot read {}: {}", file_path.display(), e);
                any_uncertifiable = true;
                continue;
            }
        };

        let cert_path = cert_toml_path(file_path);
        let rel_path = workspace_relative(file_path, &cert_path);

        let cert_text = std::fs::read_to_string(&cert_path).unwrap_or_default();

        // Handle --waive before running the scan
        if let Some(ref waive_arg) = cli.waive {
            match waive_arg.split_once(':') {
                Some((cid, reason)) => {
                    match write_waiver(&cert_path, &rel_path, cid.trim(), reason.trim()) {
                        Ok(()) => println!("Waiver recorded: {} = {:?} in {}", cid.trim(), reason.trim(), cert_path.display()),
                        Err(e) => eprintln!("cert_check: waiver write failed: {}", e),
                    }
                }
                None => {
                    eprintln!("cert_check: --waive format is CID:reason  (e.g. B6:debug bridge code)");
                }
            }
            // Re-read with new waiver
            let cert_text = std::fs::read_to_string(&cert_path).unwrap_or_default();
            let waivers = read_waivers(&cert_text, &rel_path);
            let st = scan(&content, &ac);
            let results = evaluate(&st, &waivers);
            print_results(file_path, &results, &st);
            continue;
        }

        let waivers = read_waivers(&cert_text, &rel_path);

        // Single-pass FSE scan — O(M), independent of criterion count
        let st = scan(&content, &ac);
        let results = evaluate(&st, &waivers);

        print_results(file_path, &results, &st);

        let certifiable = results.iter().all(|r| !r.status.is_blocking());

        if cli.seal {
            if certifiable {
                match sha256_hex(file_path) {
                    Ok(sha) => {
                        let date = chrono::Local::now().format("%Y-%m-%d").to_string();
                        match write_seal(&cert_path, &rel_path, &sha, &date) {
                            Ok(()) => println!("  Sealed: {} → {}", rel_path, cert_path.display()),
                            Err(e) => eprintln!("cert_check: seal write failed: {}", e),
                        }
                    }
                    Err(e) => eprintln!("cert_check: sha256 failed: {}", e),
                }
            } else {
                eprintln!("cert_check: --seal refused: {} has blocking failures", rel_path);
                any_uncertifiable = true;
            }
        }

        if !certifiable {
            any_uncertifiable = true;
        }
    }

    if any_uncertifiable {
        std::process::exit(1);
    }
}
