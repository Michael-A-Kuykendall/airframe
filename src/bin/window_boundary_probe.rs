// Cross-platform: suppress macOS clippy 1.86+ lints
#![allow(
    unknown_lints,
    clippy::manual_is_multiple_of,
    clippy::collapsible_match
)]

use clap::{Parser, ValueEnum};
use serde_json::{json, Value};
use std::fs;
use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::path::Path;
use std::time::Duration;

#[derive(Parser, Debug)]
#[command(name = "window_boundary_probe")]
#[command(about = "Window boundary probe: retrieval accuracy vs distance")]
struct Args {
    #[arg(long, default_value = "http://127.0.0.1:8080")]
    server: String,

    #[arg(long, default_value = "artifacts/longctx/window_boundary_probe.json")]
    out: String,

    #[arg(long, default_value_t = 55555)]
    seed: u64,

    #[arg(long, default_value_t = 512)]
    dist_start: usize,

    #[arg(long, default_value_t = 2048)]
    dist_end: usize,

    #[arg(long, default_value_t = 256)]
    dist_step: usize,

    #[arg(long, default_value_t = 6)]
    trials_per_dist: usize,

    #[arg(long, default_value_t = 9000)]
    noise_total_words: usize,

    #[arg(long, default_value_t = 24)]
    answer_max_tokens: usize,

    #[arg(long, default_value_t = 120.0)]
    request_timeout_sec: f64,

    #[arg(long, default_value_t = 2)]
    max_retries: usize,

    #[arg(long, default_value_t = 0.95)]
    reliable_threshold: f64,

    #[arg(long, value_enum, default_value_t = PromptLayout::DistanceOnly)]
    layout: PromptLayout,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum PromptLayout {
    FixedTotal,
    DistanceOnly,
}

struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.state >> 32) as u32
    }

    fn choose_idx(&mut self, upper_exclusive: usize) -> usize {
        if upper_exclusive == 0 {
            0
        } else {
            (self.next_u32() as usize) % upper_exclusive
        }
    }
}

fn random_tag(rng: &mut Lcg, n: usize) -> String {
    const ALNUM: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    let mut out = String::with_capacity(n);
    for _ in 0..n {
        out.push(ALNUM[rng.choose_idx(ALNUM.len())] as char);
    }
    out
}

fn make_noise_words(count: usize, prefix: &str) -> String {
    let token = prefix.chars().next().unwrap_or('x');
    if count == 0 {
        String::new()
    } else {
        std::iter::repeat_n(token.to_string(), count)
            .collect::<Vec<_>>()
            .join(" ")
    }
}

fn build_prompt(
    distance_words: usize,
    noise_words_total: usize,
    key: &str,
    value: &str,
    layout: PromptLayout,
) -> String {
    let (left, right) = match layout {
        PromptLayout::FixedTotal => (
            noise_words_total.saturating_sub(distance_words),
            distance_words,
        ),
        PromptLayout::DistanceOnly => (0, distance_words),
    };

    let left_noise = make_noise_words(left, "L");
    let right_noise = make_noise_words(right, "R");

    let fact = format!("\nMEMORY_FACT: {} = {}\n", key, value);
    let question = format!(
        "\nQuestion: What is the exact value for {}? Answer with value only.\nAnswer:",
        key
    );

    format!("{}{}{}{}", left_noise, fact, right_noise, question)
}

fn parse_server_url(server: &str) -> Result<(String, u16, String), String> {
    let without_scheme = server
        .strip_prefix("http://")
        .ok_or_else(|| format!("unsupported server URL (expected http://): {}", server))?;

    let mut parts = without_scheme.splitn(2, '/');
    let host_port = parts.next().unwrap_or_default();
    let path = format!("/{}", parts.next().unwrap_or_default());

    let (host, port) = if let Some((h, p)) = host_port.rsplit_once(':') {
        let parsed_port = p
            .parse::<u16>()
            .map_err(|_| format!("invalid port in server URL: {}", server))?;
        (h.to_string(), parsed_port)
    } else {
        (host_port.to_string(), 80)
    };

    if host.is_empty() {
        return Err(format!("invalid server URL host: {}", server));
    }

    Ok((host, port, if path == "/" { "/".to_string() } else { path }))
}

fn parse_chunked_body(mut body: &[u8]) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    loop {
        let line_end = body
            .windows(2)
            .position(|w| w == b"\r\n")
            .ok_or_else(|| "malformed chunked body (missing chunk-size line ending)".to_string())?;
        let size_line = std::str::from_utf8(&body[..line_end])
            .map_err(|_| "invalid utf8 in chunk-size line".to_string())?;
        let chunk_size = usize::from_str_radix(size_line.trim(), 16)
            .map_err(|_| format!("invalid chunk size: {}", size_line))?;
        body = &body[line_end + 2..];

        if chunk_size == 0 {
            break;
        }
        if body.len() < chunk_size + 2 {
            return Err("malformed chunked body (truncated chunk)".to_string());
        }
        out.extend_from_slice(&body[..chunk_size]);
        if &body[chunk_size..chunk_size + 2] != b"\r\n" {
            return Err("malformed chunked body (missing chunk terminator)".to_string());
        }
        body = &body[chunk_size + 2..];
    }
    Ok(out)
}

fn post_json_extract_text(
    server: &str,
    payload: &Value,
    timeout: Duration,
) -> Result<String, String> {
    let (host, port, path) = parse_server_url(server)?;
    let mut addrs = format!("{}:{}", host, port)
        .to_socket_addrs()
        .map_err(|e| format!("resolve failed: {}", e))?;
    let addr = addrs
        .next()
        .ok_or_else(|| format!("could not resolve address for {}", server))?;

    let mut stream =
        TcpStream::connect_timeout(&addr, timeout).map_err(|e| format!("connect failed: {}", e))?;
    stream
        .set_read_timeout(Some(timeout))
        .map_err(|e| format!("set_read_timeout failed: {}", e))?;
    stream
        .set_write_timeout(Some(timeout))
        .map_err(|e| format!("set_write_timeout failed: {}", e))?;

    let body = serde_json::to_vec(payload).map_err(|e| format!("serialize failed: {}", e))?;
    let req = format!(
        "POST {} HTTP/1.1\r\nHost: {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        path,
        host,
        body.len()
    );
    stream
        .write_all(req.as_bytes())
        .map_err(|e| format!("write headers failed: {}", e))?;
    stream
        .write_all(&body)
        .map_err(|e| format!("write body failed: {}", e))?;
    stream.flush().map_err(|e| format!("flush failed: {}", e))?;

    let mut response = Vec::new();
    stream
        .read_to_end(&mut response)
        .map_err(|e| format!("read response failed: {}", e))?;

    let split = response
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .ok_or_else(|| "malformed HTTP response (no header/body split)".to_string())?;

    let header_bytes = &response[..split];
    let body_bytes = &response[split + 4..];

    let headers = String::from_utf8(header_bytes.to_vec())
        .map_err(|_| "invalid utf8 headers in response".to_string())?;
    let mut header_lines = headers.lines();
    let status_line = header_lines
        .next()
        .ok_or_else(|| "missing HTTP status line".to_string())?;
    if !status_line.contains(" 200 ") {
        return Err(format!("non-200 response: {}", status_line));
    }

    let is_chunked = headers
        .to_ascii_lowercase()
        .contains("transfer-encoding: chunked");
    let decoded_body = if is_chunked {
        parse_chunked_body(body_bytes)?
    } else {
        body_bytes.to_vec()
    };

    let data: Value =
        serde_json::from_slice(&decoded_body).map_err(|e| format!("json decode failed: {}", e))?;
    Ok(data
        .get("text")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string())
}

fn ask(
    server: &str,
    prompt: &str,
    seed: u64,
    max_tokens: usize,
    request_timeout_sec: f64,
    max_retries: usize,
) -> String {
    let payload = json!({
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
        "seed": seed,
        "stream": false
    });

    let timeout = Duration::from_secs_f64(request_timeout_sec.max(1.0));

    for attempt in 0..=max_retries {
        match post_json_extract_text(server, &payload, timeout) {
            Ok(text) => return text,
            Err(err) => {
                if attempt >= max_retries {
                    println!("request_failed: attempts={} error={}", attempt + 1, err);
                    return String::new();
                }
                std::thread::sleep(Duration::from_millis(500));
            }
        }
    }
    String::new()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let mut rng = Lcg::new(args.seed);
    let mut rows = Vec::new();

    let mut d = args.dist_start;
    while d <= args.dist_end {
        let mut hits = 0usize;
        let mut trial_rows = Vec::new();

        for t in 0..args.trials_per_dist {
            let key = format!("K_{}_{}_{}", d, t, random_tag(&mut rng, 6));
            let value = format!("V_{}", random_tag(&mut rng, 10));
            let prompt = build_prompt(d, args.noise_total_words, &key, &value, args.layout);

            let pred = ask(
                &args.server,
                &prompt,
                args.seed,
                args.answer_max_tokens,
                args.request_timeout_sec,
                args.max_retries,
            );

            let exact = pred == value;
            if exact {
                hits += 1;
            }

            trial_rows.push(json!({
                "distance_words": d,
                "trial": t,
                "key": key,
                "expected": value,
                "pred": pred,
                "exact": exact
            }));
        }

        let acc = if args.trials_per_dist == 0 {
            0.0
        } else {
            hits as f64 / args.trials_per_dist as f64
        };

        println!(
            "distance={}  hits={}/{}  acc={:.3}",
            d, hits, args.trials_per_dist, acc
        );

        rows.push(json!({
            "distance_words": d,
            "trials": args.trials_per_dist,
            "hits": hits,
            "accuracy": acc,
            "details": trial_rows
        }));

        if args.dist_step == 0 {
            break;
        }
        match d.checked_add(args.dist_step) {
            Some(next) => d = next,
            None => break,
        }
    }

    let accuracies: Vec<f64> = rows
        .iter()
        .filter_map(|r| r.get("accuracy").and_then(|a| a.as_f64()))
        .collect();
    let avg_accuracy = if accuracies.is_empty() {
        0.0
    } else {
        accuracies.iter().sum::<f64>() / accuracies.len() as f64
    };

    let reliable_distances: Vec<usize> = rows
        .iter()
        .filter_map(|r| {
            let acc = r.get("accuracy")?.as_f64()?;
            let dist = r.get("distance_words")?.as_u64()? as usize;
            (acc >= args.reliable_threshold).then_some(dist)
        })
        .collect();

    let fuzzy_distances: Vec<usize> = rows
        .iter()
        .filter_map(|r| {
            let acc = r.get("accuracy")?.as_f64()?;
            let dist = r.get("distance_words")?.as_u64()? as usize;
            (acc < args.reliable_threshold).then_some(dist)
        })
        .collect();

    let summary = json!({
        "suite": "window-boundary-probe",
        "server": args.server,
        "seed": args.seed,
        "distance_metric": "approximate word-distance proxy",
        "dist_start": args.dist_start,
        "dist_end": args.dist_end,
        "dist_step": args.dist_step,
        "trials_per_dist": args.trials_per_dist,
        "noise_total_words": args.noise_total_words,
        "layout": match args.layout {
            PromptLayout::FixedTotal => "fixed_total",
            PromptLayout::DistanceOnly => "distance_only",
        },
        "reliable_threshold": args.reliable_threshold,
        "avg_accuracy": avg_accuracy,
        "max_reliable_distance_words": reliable_distances.iter().max().copied(),
        "first_fuzzy_distance_words": fuzzy_distances.iter().min().copied(),
        "rows": rows
    });

    let out_path = Path::new(&args.out);
    if let Some(parent) = out_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(out_path, serde_json::to_string_pretty(&summary)?)?;
    println!("Wrote {}", out_path.display());

    Ok(())
}
