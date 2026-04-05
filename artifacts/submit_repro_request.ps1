$Body = Get-Content $args[0] -Raw
$Url = $args[1]
$OutPath = $args[2]

Invoke-RestMethod -Uri $Url -Method Post -ContentType 'application/json' -Body $Body |
    ConvertTo-Json -Compress |
    Set-Content $OutPath