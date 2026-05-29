"""
template_health_check.py  —  End-to-end template pipeline validation

These tests detect template misconfiguration, not model intelligence.
A broken template produces characteristic failure signatures:
  - Model echoes the raw template tags (<|user|>, [INST], etc.)
  - Model ignores the system message entirely
  - Model continues as the *user* instead of the assistant
  - Model generates the next user turn instead of an assistant reply
  - Response is incoherent / garbage (prompt wasn't understood)

Usage:
  python scripts/template_health_check.py --port 8080 --model Llama-3.2-1B-Instruct-Q4_K_M
  python scripts/template_health_check.py --port 8081 --model TinyLlama-1.1B-Chat-v1.0.Q4_0
"""
import argparse, json, re, urllib.request, time

# ─────────────────────────────────────────────────────────────────────────────
# Test definitions
# Each test: (name, messages, system, check_fn, description)
# check_fn(response_text) -> (pass: bool, note: str)
# ─────────────────────────────────────────────────────────────────────────────

def no_raw_tags(text):
    """Template tags should never appear in generated output."""
    tags = ["<|user|>", "<|system|>", "<|assistant|>", "[INST]", "[/INST]",
            "<<SYS>>", "<</SYS>>", "<|im_start|>", "<|im_end|>",
            "<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>",
            "<|eot_id|>", "<bos>", "<eos>", "<s>", "</s>"]
    found = [t for t in tags if t in text]
    if found:
        return False, f"raw template tags in output: {found}"
    return True, "clean"

def is_coherent(text):
    """Output should be English words, not gibberish or repeated tokens."""
    if len(text.strip()) < 3:
        return False, "response too short / empty"
    # Repeated token hallucination: same 3+ char token repeated 4+ times
    m = re.search(r'(.{3,})\1{3,}', text)
    if m:
        return False, f"repetition loop: '{m.group(1)}...'"
    return True, "ok"

def contains_word(word):
    def check(text):
        if word.lower() in text.lower():
            return True, f"found '{word}'"
        return False, f"'{word}' not found in: {text[:80]!r}"
    return check

def does_not_contain(word):
    def check(text):
        if word.lower() in text.lower():
            return False, f"should not contain '{word}' but did: {text[:80]!r}"
        return True, "ok"
    return check

def system_message_respected(expected_persona_word):
    """Check that system message actually changed behavior."""
    def check(text):
        ok, note = contains_word(expected_persona_word)(text)
        if ok:
            return True, f"system message respected (found '{expected_persona_word}')"
        return False, f"system message ignored — {note}"
    return check

def check_all(*fns):
    def check(text):
        for fn in fns:
            ok, note = fn(text)
            if not ok:
                return False, note
        return True, "all checks passed"
    return check

# ─────────────────────────────────────────────────────────────────────────────

TESTS = [
    # name, system, messages, max_tokens, check_fn, what_failure_means
    (
        "no-template-tags-leak",
        None,
        [{"role": "user", "content": "Say hello."}],
        30,
        check_all(no_raw_tags, is_coherent),
        "Template tags are leaking into model output — template is being passed as literal text",
    ),
    (
        "assistant-role-assumed",
        None,
        [{"role": "user", "content": "Respond with exactly the word: READY"}],
        10,
        check_all(no_raw_tags, contains_word("READY")),
        "Model not taking the assistant role — add_generation_prompt may be broken",
    ),
    (
        "system-message-injected",
        "You are a geography expert. Always mention the word GEOGRAPHY in every response.",
        [{"role": "user", "content": "Tell me something interesting."}],
        60,
        check_all(no_raw_tags, system_message_respected("GEOGRAPHY")),
        "System message is not reaching the model — system role template path broken",
    ),
    (
        "multi-turn-context-preserved",
        None,
        [
            {"role": "user",      "content": "Remember the secret word: BANANA"},
            {"role": "assistant", "content": "Understood, I will remember BANANA."},
            {"role": "user",      "content": "What was the secret word I told you?"},
        ],
        20,
        check_all(no_raw_tags, contains_word("BANANA")),
        "Multi-turn history not being rendered — conversation loop in template broken",
    ),
    (
        "no-user-turn-continuation",
        None,
        [{"role": "user", "content": "What color is the sky?"}],
        30,
        check_all(
            no_raw_tags,
            is_coherent,
            # Model should answer, not continue with another question
            does_not_contain("What color"),
        ),
        "Model is generating the next user turn instead of answering — EOS or add_generation_prompt broken",
    ),
    (
        "factual-recall-coherent",
        None,
        [{"role": "user", "content": "What is the capital of Japan?"}],
        20,
        check_all(no_raw_tags, contains_word("Tokyo")),
        "Basic factual recall failing — likely incoherent output, model not understanding prompt",
    ),
    (
        "instruction-format-obedience",
        None,
        [{"role": "user", "content": "List exactly two colors. Use a numbered list. Nothing else."}],
        40,
        check_all(no_raw_tags, is_coherent, contains_word("1.")),
        "Model not following formatting instructions — instruction-tuning not activated (template wrong)",
    ),
    (
        "eos-stops-generation",
        None,
        [{"role": "user", "content": "Say only the word YES and nothing else."}],
        50,
        check_all(
            no_raw_tags,
            contains_word("YES"),
        ),
        "EOS handling broken — model may not be stopping or may be ignoring the instruction",
    ),
]

# ─────────────────────────────────────────────────────────────────────────────

def ask(port, model, system, messages, max_tokens):
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0}
    if system:
        payload["messages"] = [{"role": "system", "content": system}] + messages
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"http://localhost:{port}/v1/chat/completions",
        data=body, headers={"Content-Type": "application/json"}, method="POST",
    )
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=120) as resp:
        elapsed = time.monotonic() - t0
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"], round(elapsed * 1000)

def run(port, model):
    passed = 0
    results = []
    print(f"\n{'═'*68}")
    print(f"  Template Health Check — {model}")
    print(f"  Port: {port}")
    print(f"{'═'*68}")
    for name, system, messages, max_tokens, check_fn, failure_meaning in TESTS:
        try:
            text, ms = ask(port, model, system, messages, max_tokens)
            ok, note = check_fn(text)
        except Exception as e:
            text, ms, ok, note = f"ERR:{e}", 0, False, str(e)
        mark = "PASS" if ok else "FAIL"
        passed += ok
        results.append({"test": name, "ok": ok, "note": note, "raw": text, "ms": ms})
        print(f"\n  [{mark}] {name}")
        print(f"         response : {text[:90]!r}")
        print(f"         note     : {note}")
        if not ok:
            print(f"         DIAGNOSIS: {failure_meaning}")
    print(f"\n{'─'*68}")
    print(f"  Result: {passed}/{len(TESTS)} passed", "✓ HEALTHY" if passed == len(TESTS) else "⚠ ISSUES DETECTED")
    print(f"{'─'*68}\n")
    return results

if __name__ == "__main__":
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument("--port",  type=int, default=8080)
    ap.add_argument("--model", default="Llama-3.2-1B-Instruct-Q4_K_M")
    ap.add_argument("--out",   default=None)
    args = ap.parse_args()
    results = run(args.port, args.model)
    if args.out:
        out_path = args.out if os.path.isabs(args.out) else os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.out
        )
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({"model": args.model, "results": results}, f, indent=2)
        print(f"Results saved → {out_path}")
