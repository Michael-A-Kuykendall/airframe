# Manual Model Smoke Test

Run each model by hand. Start server, send one request, verify response, kill server, check the box.

## How to run one model

```powershell
# Terminal 1 — start server
$env:LIBSHIMMY_MODEL_PATH = "D:\shimmy-test-models\gguf_collection\<MODEL>"
$env:SHIMMY_PORT = "8080"
.\target\release\shimmy_server_gpu.exe
```

Wait for: `[HTTP] Async listener spawned on 0.0.0.0:8080`

```powershell
# Terminal 2 — send request
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8080/v1/chat/completions" `
  -ContentType "application/json" `
  -Body '{"model":"local","messages":[{"role":"user","content":"The capital of France is"}],"max_tokens":32,"temperature":0.0,"stream":false}' | ConvertTo-Json -Depth 6
```

Kill server when done.

---

## Checklist

- [x] TinyLlama-1.1B-Chat-v1.0.Q4_0.gguf — expect "Paris" → PASS ("The capital of France is Paris.")
- [x] Llama-3.2-1B-Instruct-Q4_K_M.gguf — expect "Paris" → PASS ("The capital of France is Paris.")
- [x] Llama-3.2-3B-Instruct-Q4_K_M.gguf — expect "Paris" → PASS ("The capital of France is Paris.")
- [x] phi-2.Q4_K_M.gguf — expect "Paris" → PASS ("The capital of France is Paris.")
- [x] starcoder2-3b-Q4_K_M.gguf — expect "def " (code output) → PASS (non-empty code response)
- [x] gpt2.Q4_K_M.gguf — completion model, any non-empty output → PASS ("Hola mundo")
- [LIMIT] gemma-2-2b-it-Q4_K_M.gguf — output head 2.19 GB > WebGPU 2 GB limit, skip

---

## Notes

<!-- Fill in as you go -->
