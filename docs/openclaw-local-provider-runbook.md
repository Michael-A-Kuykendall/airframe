# OpenClaw Local Provider Runbook

This branch is the fastest path to using this machine as a private provider for OpenClaw on another Windows machine.

## Recommendation

Use `shimmy_integration` as the provider surface first.

Reason:

- it already exposes OpenAI-compatible endpoints
- OpenClaw is most likely to accept that surface with minimal work
- the current Airframe repro server is useful for runtime proofing, but it is not the shortest path to a provider integration

Treat the Airframe work on this branch as the engine spike and treat `shimmy_integration` as the provider shell.

## What Is On This Branch

- provider run tasks in `.vscode/tasks.json`
- a Windows smoke test in `scripts/openclaw_provider_smoke_test.ps1`
- the integration framing docs in `docs/shimmy-airframe-release-strategy.md`
- the seam inventory in `docs/shimmy-airframe-seam-map.md`

## Fastest Same-Machine Path

1. Open this repo in VS Code.
2. Run the task `Run Shimmy OpenClaw Provider`.
3. Enter the absolute path to your GGUF file when prompted.
4. Keep the default bind address `127.0.0.1:11435`.
5. Run the task `Smoke Test Shimmy OpenClaw Provider`.
6. Confirm that the script prints a model id and a short completion.

At that point the provider endpoint is:

- base URL: `http://127.0.0.1:11435/v1`
- api key: any placeholder string
- model: whatever appears in `/v1/models`

## Fastest Cross-Machine Path

Use this when OpenClaw runs on a different Windows machine on the same LAN.

1. Run the task `Run Shimmy OpenClaw Provider`.
2. Use the GGUF path on the serving machine.
3. Set the bind address to `0.0.0.0:11435`.
4. Allow inbound TCP `11435` in Windows Firewall on the serving machine.
5. Find the serving machine's LAN IP with `ipconfig`.
6. Run the smoke test locally first against `http://127.0.0.1:11435`.
7. From the OpenClaw machine, use `http://SERVING_MACHINE_IP:11435/v1` as the provider base URL.

OpenClaw-side settings should be:

- provider type: OpenAI-compatible
- base URL: `http://SERVING_MACHINE_IP:11435/v1`
- api key: any non-empty placeholder
- model: first model returned by `GET /v1/models`

## Direct Commands

If you do not want to use VS Code tasks, run this from the repo root:

```powershell
cargo run --release --manifest-path shimmy_integration/Cargo.toml -- --gpu-backend auto serve --bind 127.0.0.1:11435 --model-path "C:\path\to\your-model.gguf"
```

For LAN access:

```powershell
cargo run --release --manifest-path shimmy_integration/Cargo.toml -- --gpu-backend auto serve --bind 0.0.0.0:11435 --model-path "C:\path\to\your-model.gguf"
```

Smoke test:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\openclaw_provider_smoke_test.ps1 -BaseUrl http://127.0.0.1:11435
```

## Expected Verification

The following should work before touching OpenClaw:

```powershell
Invoke-RestMethod http://127.0.0.1:11435/v1/models
```

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:11435/v1/chat/completions -ContentType application/json -Body '{"model":"REPLACE_ME","messages":[{"role":"user","content":"Say hello."}],"max_tokens":32}'
```

## Architecture Direction

If the immediate goal is speed, stop at the OpenAI-compatible provider shell and prove OpenClaw connectivity.

If that works, the next engineering move is:

1. keep Shimmy's OpenAI API layer intact
2. replace or augment Shimmy's inference backend with Airframe behind the engine seam
3. leave OpenClaw pointed at the same provider URL while the engine changes underneath

That keeps the client contract stable while the runtime evolves.

## Minimum Handoff For The Other Machine

The other machine only needs:

1. this branch
2. Rust toolchain if building from source
3. a GGUF path on the serving machine
4. the chosen bind address
5. the final provider URL for OpenClaw

## Notes

- Keep the provider bound to `127.0.0.1` unless you explicitly need LAN access.
- If port `11435` is busy, choose another port and update the OpenClaw base URL to match.
- The current Airframe `shimmy_server_gpu` binary is still useful for runtime experimentation, but it is not the primary provider surface for OpenClaw.