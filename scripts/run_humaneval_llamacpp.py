import argparse
import gzip
import json
import subprocess
import sys
from pathlib import Path


def default_problem_file():
    repo_root = Path(__file__).resolve().parents[1]
    vendored = repo_root / "vendor" / "human-eval" / "data" / "HumanEval.jsonl.gz"
    if vendored.exists():
        return vendored
    return None


def load_problems(problem_file):
    with gzip.open(problem_file, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


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


def run_generation(problem, args):
    prompt = problem["prompt"]
    cmd = [
        str(args.llama_cli),
        "-m",
        str(args.model),
        "--no-display-prompt",
        "--simple-io",
        "--log-disable",
        "--temp",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--repeat-penalty",
        str(args.repetition_penalty),
        "--seed",
        str(args.seed),
        "-n",
        str(args.max_tokens),
        "-p",
        prompt,
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    return truncate_completion(prompt, result.stdout)


def main():
    parser = argparse.ArgumentParser(description="Run HumanEval against stock llama.cpp")
    parser.add_argument("--llama-cli", required=True, help="Path to llama-cli executable")
    parser.add_argument("--model", required=True, help="Path to GGUF model")
    parser.add_argument("--output", default="artifacts/humaneval_llamacpp_samples.jsonl")
    parser.add_argument("--problem-file", default=None, help="Path to HumanEval.jsonl.gz")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N HumanEval tasks")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.llama_cli = Path(args.llama_cli)
    args.model = Path(args.model)
    problem_file = Path(args.problem_file) if args.problem_file else default_problem_file()
    if problem_file is None:
        raise SystemExit("No HumanEval problem file found")

    problems = load_problems(problem_file)
    if args.limit is not None:
        problems = problems[: args.limit]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    samples = []
    for index, problem in enumerate(problems, start=1):
        task_id = problem["task_id"]
        print(f"[{index}/{len(problems)}] submitting {task_id}", flush=True)
        try:
            completion = run_generation(problem, args)
        except subprocess.CalledProcessError as exc:
            print(exc.stdout, file=sys.stderr)
            print(exc.stderr, file=sys.stderr)
            raise

        samples.append({"task_id": task_id, "completion": completion})
        with output_path.open("w", encoding="utf-8") as handle:
            for sample in samples:
                handle.write(json.dumps(sample) + "\n")

    print(f"wrote {len(samples)} samples to {output_path}", flush=True)


if __name__ == "__main__":
    main()