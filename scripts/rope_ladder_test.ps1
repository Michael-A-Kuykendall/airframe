param(
    [string]$ServerExe   = "C:\Users\micha\repos\airframe\target\release\shimmy_server_gpu.exe",
    [string]$BaseUrl     = "http://127.0.0.1:8080",
    [string]$ResultsOut  = "C:\Users\micha\repos\airframe\artifacts\rope_ladder_results.json",
    [int]$ServerStartSec = 120,
    [int]$PollSec        = 180,
    [switch]$SkipBaseline
)

$ErrorActionPreference = "Stop"
$ProgressPreference    = "SilentlyContinue"

# ---------------------------------------------------------------------------
# Ladder definition
# Each rung: max_ctx, rope_scale (auto=0 means 2048/max_ctx), needle_offset
# (approx token position where secret is planted), filler_target (tokens of
# padding to push the needle deep into the context), expected_keyword
# ---------------------------------------------------------------------------
$Ladder = @(
    [pscustomobject]@{ max_ctx=2048;  rope_scale=1.0;    needle_offset=100; filler_target=1600; label="2048-baseline" }
    [pscustomobject]@{ max_ctx=4096;  rope_scale=0.0;    needle_offset=100; filler_target=3600; label="4096-linear"   }
    [pscustomobject]@{ max_ctx=8192;  rope_scale=0.0;    needle_offset=100; filler_target=7500; label="8192-linear"   }
    [pscustomobject]@{ max_ctx=16384; rope_scale=0.0;    needle_offset=100; filler_target=15000; label="16384-linear" }
    [pscustomobject]@{ max_ctx=32768; rope_scale=0.0;    needle_offset=100; filler_target=30000; label="32768-linear" }
)

# The secret that must survive retrieval
$SecretCode = "XYLOPHONE-7743"
$ExtractQuestion = "What is the secret code mentioned earlier in this document? Reply with ONLY the code, nothing else."

# Filler sentence repeated to pad — ~15 tokens each repetition
$FillerSentence = "The mountain range extends across the northern hemisphere, providing habitat for numerous species of birds and mammals. "

# ---------------------------------------------------------------------------
function Wait-ServerReady {
    param([string]$Url, [int]$MaxSec)
    $readyUrl = "$Url/api/repro/queue"
    for ($i = 0; $i -lt $MaxSec; $i++) {
        try {
            $null = Invoke-RestMethod -Method Get -Uri $readyUrl -ErrorAction Stop
            return $true
        } catch {
            Start-Sleep -Seconds 1
        }
    }
    return $false
}

function Submit-Job {
    param([string]$Url, [hashtable]$Req)
    $body = $Req | ConvertTo-Json -Compress -Depth 5
    $resp = Invoke-RestMethod -Method Post -Uri "$Url/" -ContentType "application/json" -Body $body -ErrorAction Stop
    return $resp.job_id
}

function Poll-Job {
    param([string]$Url, [string]$JobId, [int]$MaxSec)
    $statusUrl = "$Url/api/repro/job-status?job_id=$JobId"
    for ($i = 0; $i -lt $MaxSec; $i++) {
        $s = Invoke-RestMethod -Method Get -Uri $statusUrl -ErrorAction Stop
        if ($s.status -eq "completed" -or $s.status -eq "failed") {
            return $s
        }
        Start-Sleep -Seconds 1
    }
    return $null
}

function Kill-Server {
    $procs = Get-Process -Name "shimmy_server_gpu" -ErrorAction SilentlyContinue
    if ($procs) {
        $procs | Stop-Process -Force
        Start-Sleep -Seconds 2
    }
}

function Start-Server {
    param([int]$MaxCtx, [float]$RopeScale)

    Kill-Server

    $env:SHIMMY_MAX_CTX    = [string]$MaxCtx
    if ($RopeScale -gt 0.0) {
        $env:SHIMMY_ROPE_SCALE = [string]$RopeScale
    } else {
        # auto: linear = 2048 / max_ctx
        $env:SHIMMY_ROPE_SCALE = [string](2048.0 / $MaxCtx)
    }

    Write-Host "  Starting server: ctx=$MaxCtx scale=$($env:SHIMMY_ROPE_SCALE)"
    $proc = Start-Process `
        -FilePath $ServerExe `
        -PassThru `
        -RedirectStandardError "C:\Users\micha\repos\airframe\artifacts\rope_ladder_server_stderr.txt" `
        -NoNewWindow
    return $proc
}

function Build-NeedlePrompt {
    param([int]$FillerTarget, [string]$FillerSentence, [string]$SecretCode, [string]$Question)

    # Rough token estimate: 1 token ≈ 4 chars for this filler
    $charsNeeded = $FillerTarget * 4
    $reps = [int][Math]::Ceiling($charsNeeded / $FillerSentence.Length)

    $filler = $FillerSentence * $reps

    # Structure: opening + needle sentence + filler + question
    $prompt  = "This is a long document for testing context retention.`n`n"
    $prompt += "SECRET CODE: $SecretCode`n`n"
    $prompt += $filler.Substring(0, [Math]::Min($filler.Length, $charsNeeded))
    $prompt += "`n`n$Question"

    return $prompt
}

# ---------------------------------------------------------------------------
$results = @()

foreach ($rung in $Ladder) {
    if ($SkipBaseline -and $rung.label -eq "2048-baseline") {
        Write-Host "`n[SKIP] $($rung.label)"
        continue
    }

    Write-Host "`n=== RUNG: $($rung.label) (ctx=$($rung.max_ctx)) ==="

    $proc = Start-Server -MaxCtx $rung.max_ctx -RopeScale $rung.rope_scale

    $ready = Wait-ServerReady -Url $BaseUrl -MaxSec $ServerStartSec
    if (-not $ready) {
        Write-Warning "  Server did not start in $ServerStartSec s — skipping rung"
        Kill-Server
        $results += [pscustomobject]@{
            label        = $rung.label
            max_ctx      = $rung.max_ctx
            rope_scale   = if ($rung.rope_scale -gt 0) { $rung.rope_scale } else { 2048.0 / $rung.max_ctx }
            filler_target = $rung.filler_target
            pass         = $false
            found_keyword = $false
            response_text = $null
            error        = "server_start_timeout"
        }
        continue
    }
    Write-Host "  Server ready."

    # Log what the server printed as metadata
    Start-Sleep -Seconds 1

    $prompt = Build-NeedlePrompt `
        -FillerTarget   $rung.filler_target `
        -FillerSentence $FillerSentence `
        -SecretCode     $SecretCode `
        -Question       $ExtractQuestion

    $promptChars = $prompt.Length
    $approxTokens = [int]($promptChars / 4)
    Write-Host "  Prompt: ~$approxTokens tokens ($promptChars chars)"

    $reqBody = @{
        task        = "needle"
        prompt      = $prompt
        prompt_mode = "raw"
        max_tokens  = 32
        temperature = 0.0
        top_p       = 1.0
        seed        = 42
        stream      = $false
    }

    $err = $null
    $foundKeyword = $false
    $responseText = $null

    try {
        $jobId = Submit-Job -Url $BaseUrl -Req $reqBody
        Write-Host "  Job submitted: $jobId"

        $status = Poll-Job -Url $BaseUrl -JobId $jobId -MaxSec $PollSec

        if ($null -eq $status) {
            $err = "poll_timeout"
        } elseif ($status.status -ne "completed") {
            $err = "job_status:$($status.status) error:$($status.error)"
        } else {
            $responseText = [string]$status.result.text
            $foundKeyword = $responseText -like "*$SecretCode*"
        }
    } catch {
        $err = "exception:$($_.Exception.Message)"
    }

    $pass = (-not $err) -and $foundKeyword

    $rungResult = [pscustomobject]@{
        label         = $rung.label
        max_ctx       = $rung.max_ctx
        rope_scale    = if ($rung.rope_scale -gt 0) { $rung.rope_scale } else { 2048.0 / $rung.max_ctx }
        filler_target = $rung.filler_target
        approx_prompt_tokens = $approxTokens
        pass          = $pass
        found_keyword = $foundKeyword
        response_text = $responseText
        error         = $err
    }
    $results += $rungResult

    $status_str = if ($pass) { "PASS" } elseif ($foundKeyword) { "PASS(no-err)" } else { "FAIL" }
    Write-Host "  Result: $status_str  found=$foundKeyword  err=$err"
    Write-Host "  Response: $responseText"

    # Bail out early if we've lost retrieval — no point running wider ctx
    if (-not $pass -and -not $foundKeyword) {
        Write-Host "`n  Retrieval lost at $($rung.label) — stopping ladder."
        Kill-Server
        break
    }

    Kill-Server
}

# ---------------------------------------------------------------------------
# Summary
Write-Host "`n=== SUMMARY ==="
$results | ForEach-Object {
    $marker = if ($_.pass) { "[PASS]" } else { "[FAIL]" }
    Write-Host "$marker  $($_.label)  ctx=$($_.max_ctx)  scale=$($_.rope_scale)  filler_tokens=~$($_.filler_target)  found=$($_.found_keyword)"
}

$results | ConvertTo-Json -Depth 5 | Set-Content -Path $ResultsOut -Encoding UTF8
Write-Host "`nResults written to: $ResultsOut"
