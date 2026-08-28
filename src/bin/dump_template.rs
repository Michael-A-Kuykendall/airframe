//! Audit helper: dump a model's GGUF chat_template (and arch) using the real
//! airframe metadata parser (NOT hand-parsing). For the templating audit only.
use std::env;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use airframe::backend::bindless::metadata::BindlessMetadata;

fn main() {
    for path in env::args().skip(1) {
        let p = Path::new(&path);
        if !p.is_file() {
            println!("{} -> MISSING", path);
            continue;
        }
        let file = File::open(p).expect("open");
        let mut reader = BufReader::new(file);
        let meta = BindlessMetadata::new(&mut reader);
        let spec = meta.to_model_spec();
        let arch = meta
            .gguf_metadata
            .get("general.architecture")
            .map(|v| match v {
                airframe::core::spec::GgufValue::String(s) => s.clone(),
                _ => format!("{:?}", v),
            })
            .unwrap_or_default();
        let tmpl = spec.chat_template.as_deref().unwrap_or("NONE");
        let trimmed = if tmpl.len() > 200 {
            format!("{}...", &tmpl[..200])
        } else {
            tmpl.to_string()
        };
        println!(
            "=== {} ===\n  arch: {}\n  chat_template ({}): {}",
            p.file_name().unwrap_or_default().to_string_lossy(),
            arch,
            spec.chat_template.as_ref().map(|s| s.len()).unwrap_or(0),
            trimmed
        );
    }
}
