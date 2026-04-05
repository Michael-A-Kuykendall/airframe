param(
    [string]$BaseUrl = "http://127.0.0.1:8080",
    [int]$MaxTokens = 300,
    [int]$PollSeconds = 240,
    [string]$OutputRoot = "C:\Users\micha\repos\airframe\artifacts\card_ab"
)

$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------
# Constraint suite: each case has a prompt, card fields, and a list of checks.
# Each check is a regex that must match (or must NOT match) the response text.
# ---------------------------------------------------------------------------
$Suite = @(
    @{
        Name = "parse_port"
        Prompt = "Write a Rust function named parse_port that trims whitespace, rejects empty input with an Err, rejects non-numeric input with an Err, rejects values outside 1..=65535 with an Err, does not panic, and includes unit tests. Output only Rust code."
        Goal = "parse_port"
        Must = @("reject empty input", "reject out-of-range values", "include unit tests")
        MustNot = @("panic on bad input")
        Checks = @(
            @{ Desc = "fn parse_port defined";   Regex = "fn\s+parse_port";        Require = $true  }
            @{ Desc = "returns Result";           Regex = "Result<";                Require = $true  }
            @{ Desc = "has unit tests";           Regex = "#\[test\]|#\[cfg\(test"; Require = $true  }
            @{ Desc = "no bare unwrap on parse";  Regex = "\.parse\(\)\.unwrap\("; Require = $false }
            @{ Desc = "empty check present";      Regex = "is_empty\(\)|\.trim\(\)\.is_empty\(\)|\.len\(\)\s*==\s*0"; Require = $true }
        )
    },
    @{
        Name = "clamp_u8"
        Prompt = "Write a Rust function named clamp_u8 that takes an i32 and clamps it to the range 0..=255, returning a u8. It must not use the standard library clamp method. Include at least two unit tests covering boundary values. Output only Rust code."
        Goal = "clamp_u8"
        Must = @("no std clamp", "test boundary 0", "test boundary 255")
        MustNot = @("use std::cmp::clamp", "use .clamp(")
        Checks = @(
            @{ Desc = "fn clamp_u8 defined";         Regex = "fn\s+clamp_u8";           Require = $true  }
            @{ Desc = "returns u8";                  Regex = "->\s*u8";                  Require = $true  }
            @{ Desc = "has unit tests";              Regex = "#\[test\]|#\[cfg\(test";   Require = $true  }
            @{ Desc = "no .clamp( call";             Regex = "\.clamp\(";               Require = $false }
            @{ Desc = "boundary 0 or 255 in tests";  Regex = "0\s*,|255\s*[,\)]|assert.*0|assert.*255"; Require = $true }
        )
    },
    @{
        Name = "count_vowels"
        Prompt = "Write a Rust function named count_vowels that takes a &str and returns the count of ASCII vowels (a, e, i, o, u, case-insensitive) as a usize. It must not allocate a new String. Include at least one unit test. Output only Rust code."
        Goal = "count_vowels"
        Must = @("no String allocation", "count case-insensitive", "include unit test")
        MustNot = @("to_lowercase().collect into String", "to_string()")
        Checks = @(
            @{ Desc = "fn count_vowels defined";  Regex = "fn\s+count_vowels";         Require = $true  }
            @{ Desc = "takes &str";               Regex = "&str";                       Require = $true  }
            @{ Desc = "returns usize";            Regex = "->\s*usize";                 Require = $true  }
            @{ Desc = "has unit tests";           Regex = "#\[test\]|#\[cfg\(test";    Require = $true  }
            @{ Desc = "no to_string() call";      Regex = "\.to_string\(\)";           Require = $false }
        )
    },
    @{
        Name = "is_palindrome"
        Prompt = "Write a Rust function named is_palindrome that takes a &str and returns true if the string is a palindrome ignoring ASCII case and non-alphanumeric characters. Do not use collect into a Vec or String for the comparison. Include unit tests for empty string, single char, and a mixed-case palindrome. Output only Rust code."
        Goal = "is_palindrome"
        Must = @("ignore case", "ignore non-alphanumeric", "test empty string", "test mixed-case palindrome")
        MustNot = @("collect into Vec or String for comparison")
        Checks = @(
            @{ Desc = "fn is_palindrome defined"; Regex = "fn\s+is_palindrome";       Require = $true  }
            @{ Desc = "returns bool";             Regex = "->\s*bool";                 Require = $true  }
            @{ Desc = "has unit tests";           Regex = "#\[test\]|#\[cfg\(test";   Require = $true  }
            @{ Desc = "is_alphanumeric present";  Regex = "is_alphanumeric|is_ascii_alphanumeric"; Require = $true }
            @{ Desc = "to_ascii_lowercase used";  Regex = "to_ascii_lowercase|to_lowercase|eq_ignore_ascii_case"; Require = $true }
        )
    },
    @{
        Name = "safe_divide"
        Prompt = "Write a Rust function named safe_divide that takes two f64 values and returns Ok(f64) with the quotient, or Err(&'static str) with message 'division by zero' if the divisor is zero. Do not use unwrap or expect. Include unit tests for the zero case and a normal case. Output only Rust code."
        Goal = "safe_divide"
        Must = @("return Err for zero divisor", "error message 'division by zero'", "include unit tests")
        MustNot = @("panic", "unwrap", "expect")
        Checks = @(
            @{ Desc = "fn safe_divide defined";        Regex = "fn\s+safe_divide";              Require = $true  }
            @{ Desc = "returns Result";                Regex = "Result<";                        Require = $true  }
            @{ Desc = "division by zero message";      Regex = "division by zero";               Require = $true  }
            @{ Desc = "has unit tests";                Regex = "#\[test\]|#\[cfg\(test";        Require = $true  }
            @{ Desc = "no unwrap/expect";              Regex = "\.unwrap\(\)|\.expect\(";       Require = $false }
        )
    }
)

$Modes = @("off", "shadow", "on")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
function Wait-ServerReady {
    param([string]$Url)
    $queueUrl = "$Url/api/repro/queue"
    for ($i = 0; $i -lt 120; $i++) {
        try { $null = Invoke-RestMethod -Method Get -Uri $queueUrl; return } catch { Start-Sleep -Seconds 1 }
    }
    throw "Server not ready at $Url"
}

function Submit-AndPoll {
    param([string]$Url, [hashtable]$Body, [int]$MaxPollSeconds)
    $bodyJson = $Body | ConvertTo-Json -Depth 8
    $submit = Invoke-RestMethod -Method Post -Uri "$Url/" -ContentType "application/json" -Body $bodyJson
    if (-not $submit.job_id) { throw "No job_id returned" }
    $jobId = [string]$submit.job_id
    $statusUrl = "$Url/api/repro/job-status?job_id=$jobId"
    $maxPolls = [Math]::Ceiling($MaxPollSeconds / 5)
    for ($i = 0; $i -lt $maxPolls; $i++) {
        $status = Invoke-RestMethod -Method Get -Uri $statusUrl
        if ($status.status -eq "completed" -or $status.status -eq "failed") {
            # If policy filtered the text, fall back to partial_text for scoring purposes
            if ($status.result -and ([string]$status.result.text).Length -eq 0 -and $status.partial_text) {
                $status.result | Add-Member -NotePropertyName "scoring_text" -NotePropertyValue ([string]$status.partial_text) -Force
            } else {
                $status.result | Add-Member -NotePropertyName "scoring_text" -NotePropertyValue ([string]$status.result.text) -Force
            }
            return $status
        }
        Start-Sleep -Seconds 5
    }
    throw "Job $jobId timed out after $MaxPollSeconds seconds"
}

function Score-Output {
    param([string]$Text, [array]$Checks)
    $passed = 0
    $results = @()
    foreach ($check in $Checks) {
        $matched = ($Text -match $check.Regex)
        $ok = if ($check.Require) { $matched } else { -not $matched }
        if ($ok) { $passed++ }
        $results += [pscustomobject]@{
            Desc    = $check.Desc
            Require = $check.Require
            Matched = $matched
            Pass    = $ok
        }
    }
    return @{ Passed = $passed; Total = $Checks.Count; Results = $results }
}

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
Wait-ServerReady -Url $BaseUrl

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$runDir = Join-Path $OutputRoot $stamp
$null = New-Item -ItemType Directory -Force -Path $runDir

$allRows = @()

foreach ($case in $Suite) {
    foreach ($mode in $Modes) {
        $sessionId = "ab-$($case.Name)-$mode-$stamp"
        Write-Host "Running $($case.Name) / $mode ..." -NoNewline

        $baseBody = @{
            prompt               = $case.Prompt
            session_id           = $sessionId
            prompt_mode          = "developer"
            card_processing_mode = $mode
            max_tokens           = $MaxTokens
            temperature          = 0.0
            top_p                = 1.0
            repetition_penalty   = 1.0
            seed                 = 7777
            stream               = $false
        }
        if ($mode -ne "off") { $baseBody["task"] = "card_smoke" }

        try {
            # Warmup turn: bootstrap the card for shadow/on modes using a brief prompt.
            # This populates the stored card so that turn 2 (on mode) has something to inject.
            if ($mode -ne "off") {
                $warmupBody = $baseBody.Clone()
                $warmupBody["seed"] = 1234
                $warmupBody["max_tokens"] = 128
                $null = Submit-AndPoll -Url $BaseUrl -Body $warmupBody -MaxPollSeconds 120
            }

            # Measurement turn: this is what we score
            $body = $baseBody.Clone()
            $status = Submit-AndPoll -Url $BaseUrl -Body $body -MaxPollSeconds $PollSeconds
            $scoringText = if ($status.result -and $status.result.scoring_text) { [string]$status.result.scoring_text } else { "" }
            $text = if ($status.result -and $status.result.text) { [string]$status.result.text } else { "" }
            $score  = Score-Output -Text $scoringText -Checks $case.Checks
            $tokens = [int]$status.result.tokens_generated
            $stop   = [string]$status.result.stop_reason
            $policyStatus = if ($status.result.policy_status) { [string]$status.result.policy_status } else { "ok" }

            # Save output
            $caseDir = Join-Path $runDir "$($case.Name)_$mode"
            $null = New-Item -ItemType Directory -Force -Path $caseDir
            # Save raw model output (partial_text / scoring_text) for inspection
            Set-Content -Path (Join-Path $caseDir "response.txt") -Value $scoringText -NoNewline
            if ($text -ne $scoringText) {
                Set-Content -Path (Join-Path $caseDir "response_filtered.txt") -Value "(policy filtered: $policyStatus)" -NoNewline
            }
            $score.Results | ConvertTo-Json -Depth 5 | Set-Content -Path (Join-Path $caseDir "checks.json")

            $policyTag = if ($policyStatus -ne "ok") { " policy=$policyStatus" } else { "" }
            Write-Host " $($score.Passed)/$($score.Total) constraints  ($tokens tok, $stop$policyTag)"

            $allRows += [pscustomobject]@{
                Case             = $case.Name
                Mode             = $mode
                Tokens           = $tokens
                StopReason       = $stop
                ConstraintsPassed = $score.Passed
                ConstraintsTotal = $score.Total
                Score            = [math]::Round(100.0 * $score.Passed / $score.Total, 0)
            }
        } catch {
            Write-Host " ERROR: $_"
            $allRows += [pscustomobject]@{
                Case             = $case.Name
                Mode             = $mode
                Tokens           = 0
                StopReason       = "error"
                ConstraintsPassed = 0
                ConstraintsTotal = $case.Checks.Count
                Score            = 0
            }
        }
    }
}

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
$allRows | ConvertTo-Json -Depth 5 | Set-Content -Path (Join-Path $runDir "results.json")

Write-Host ""
Write-Host "=== A/B Results ===" -ForegroundColor Cyan
$allRows | Format-Table -AutoSize

# Per-mode aggregates
Write-Host "=== Mode Averages ===" -ForegroundColor Cyan
foreach ($mode in $Modes) {
    $modeRows = $allRows | Where-Object { $_.Mode -eq $mode }
    $avg = [math]::Round(($modeRows | Measure-Object -Property Score -Average).Average, 1)
    $tokens = [math]::Round(($modeRows | Measure-Object -Property Tokens -Average).Average, 0)
    Write-Host ("  {0,-8} avg_score={1}%  avg_tokens={2}" -f $mode, $avg, $tokens)
}

Write-Host ""
Write-Host "Run artifacts: $runDir"
