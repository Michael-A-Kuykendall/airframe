import argparse
import gzip
import http.client
import json
import socket
import sys
import time
import urllib.error
import urllib.parse
from pathlib import Path


def default_problem_file():
    repo_root = Path(__file__).resolve().parents[1]
    vendored = repo_root / "vendor" / "human-eval" / "data" / "HumanEval.jsonl.gz"
    if vendored.exists():
        return vendored
    return None


def load_problems(problem_file=None):
    try:
        from human_eval.data import read_problems
    except ImportError as exc:
        raise SystemExit(
            "human_eval is not installed. Install it first with pip install git+https://github.com/openai/human-eval.git"
        ) from exc

    if problem_file is None:
        return read_problems()
    return read_problems(str(problem_file))


def request_json(url, method, timeout, payload=None, retries=3, retry_delay=0.5):
    parsed = urllib.parse.urlsplit(url)
    connection_cls = (
        http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    )
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"

    body = None
    headers = {
        "Accept": "application/json",
        "Connection": "close",
    }
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    last_error = None
    for attempt in range(retries):
        conn = connection_cls(parsed.netloc, timeout=timeout)
        try:
            conn.request(method, path, body=body, headers=headers)
            response = conn.getresponse()
            raw = response.read()
            if response.status >= 400:
                detail = raw.decode("utf-8", errors="replace")
                raise RuntimeError(f"HTTP {response.status} from {url}: {detail}")
            return json.loads(raw.decode("utf-8"))
        except (ConnectionResetError, http.client.HTTPException, OSError, socket.timeout) as exc:
            last_error = exc
            if attempt + 1 == retries:
                break
            time.sleep(retry_delay * (attempt + 1))
        finally:
            conn.close()

    raise urllib.error.URLError(last_error)


def post_json(url, payload, timeout):
    return request_json(url, "POST", timeout, payload=payload)


def get_json(url, timeout):
    return request_json(url, "GET", timeout)


def submit_job(base_url, prompt, args):
    payload = {
        "task": "humaneval",
        "prompt": prompt,
        "prompt_mode": "raw",
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "seed": args.seed,
        "stream": False,
        "ignore_eos": False,
    }
    return post_json(f"{base_url}/", payload, args.http_timeout)


def poll_job(base_url, job_id, args):
    status_url = f"{base_url}/api/repro/job-status?{urllib.parse.urlencode({'job_id': job_id})}"
    deadline = time.time() + args.job_timeout
    while True:
        state = get_json(status_url, args.http_timeout)
        status = state.get("status")
        if status == "completed":
            result = state.get("result") or {}
            return result.get("text", "")
        if status == "failed":
            raise RuntimeError(state.get("error") or f"job {job_id} failed")
        if time.time() >= deadline:
            raise TimeoutError(f"job {job_id} timed out after {args.job_timeout} seconds")
        time.sleep(args.poll_interval)


def strip_markdown_fence(text):
    stripped = text.lstrip()
    if not stripped.startswith("```"):
        return text

    lines = stripped.splitlines()
    if not lines:
        return ""
    lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)


def truncate_completion(prompt, completion):
    text = completion.replace("\r\n", "\n")
    if text.startswith(prompt):
        text = text[len(prompt) :]

    text = strip_markdown_fence(text)

    stop_markers = [
        "\nif __name__ == '__main__':",
        '\nif __name__ == "__main__":',
        "\nclass ",
        "\ndef ",
        "\nprint(",
        "\n```",
    ]

    cut = len(text)
    for marker in stop_markers:
        idx = text.find(marker)
        if idx != -1:
            cut = min(cut, idx)

    return text[:cut].rstrip() + "\n"


def evaluate_samples(samples_path, problem_file=None):
    try:
        from human_eval.evaluation import evaluate_functional_correctness
    except ImportError as exc:
        raise SystemExit(
            "human_eval is installed incorrectly or missing evaluation module"
        ) from exc

    kwargs = {}
    if problem_file is not None:
        kwargs["problem_file"] = str(problem_file)
    return evaluate_functional_correctness(str(samples_path), k=[1], **kwargs)


def write_problem_subset(problems, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_path, "wt", encoding="utf-8") as handle:
        for problem in problems.values():
            handle.write(json.dumps(problem) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run HumanEval against the Airframe GPU server")
    parser.add_argument("--server", default="http://127.0.0.1:8080", help="Base URL for shimmy_server_gpu")
    parser.add_argument("--output", default="artifacts/humaneval_airframe_samples.jsonl")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N HumanEval tasks")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--http-timeout", type=float, default=30.0)
    parser.add_argument("--job-timeout", type=float, default=900.0)
    parser.add_argument("--problem-file", default=None, help="Path to HumanEval.jsonl.gz")
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()

    problem_file = Path(args.problem_file) if args.problem_file else default_problem_file()
    problems = load_problems(problem_file)
    items = list(problems.items())
    if args.limit is not None:
        items = items[: args.limit]
    selected_problems = dict(items)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    base_url = args.server.rstrip("/")
    samples = []

    for index, (task_id, problem) in enumerate(items, start=1):
        prompt = problem["prompt"]
        print(f"[{index}/{len(items)}] submitting {task_id}", flush=True)
        try:
            queued = submit_job(base_url, prompt, args)
            job_id = queued["job_id"]
            raw_text = poll_job(base_url, job_id, args)
            completion = truncate_completion(prompt, raw_text)
        except (urllib.error.URLError, KeyError, TimeoutError, RuntimeError) as exc:
            print(f"failed on {task_id}: {exc}", file=sys.stderr, flush=True)
            raise

        samples.append({"task_id": task_id, "completion": completion})

        with output_path.open("w", encoding="utf-8") as handle:
            for sample in samples:
                handle.write(json.dumps(sample) + "\n")

    print(f"wrote {len(samples)} samples to {output_path}", flush=True)

    if args.evaluate:
        evaluation_problem_file = problem_file
        if args.limit is not None:
            evaluation_problem_file = output_path.with_name(output_path.stem + "_problems.jsonl.gz")
            write_problem_subset(selected_problems, evaluation_problem_file)
        results = evaluate_samples(output_path, evaluation_problem_file)
        print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()