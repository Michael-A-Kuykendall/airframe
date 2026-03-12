param(
    [string]$BaseUrl = "http://127.0.0.1:11435",
    [string]$Model = ""
)

$ErrorActionPreference = "Stop"

function Get-ModelId {
    param(
        [string]$ResolvedBaseUrl
    )

    $modelsResponse = Invoke-RestMethod -Method Get -Uri "$ResolvedBaseUrl/v1/models"
    if ($modelsResponse.data -and $modelsResponse.data.Count -gt 0) {
        return $modelsResponse.data[0].id
    }

    throw "No models were returned from $ResolvedBaseUrl/v1/models"
}

$resolvedModel = if ([string]::IsNullOrWhiteSpace($Model)) {
    Get-ModelId -ResolvedBaseUrl $BaseUrl
} else {
    $Model
}

$body = @{
    model = $resolvedModel
    messages = @(
        @{
            role = "user"
            content = "Say hello in five words."
        }
    )
    max_tokens = 32
    stream = $false
} | ConvertTo-Json -Depth 8

$response = Invoke-RestMethod -Method Post -Uri "$BaseUrl/v1/chat/completions" -ContentType "application/json" -Body $body

$content = $response.choices[0].message.content

Write-Host "Provider URL : $BaseUrl"
Write-Host "Model        : $resolvedModel"
Write-Host "Response     : $content"