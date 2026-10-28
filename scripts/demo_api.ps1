$Body = @{
    question = "Which endpoint processes raw documents into chunks?"
    top_k = 5
} | ConvertTo-Json

$Response = Invoke-RestMethod `
    -Uri "http://127.0.0.1:8000/chat" `
    -Method POST `
    -ContentType "application/json" `
    -Body $Body

$Response | ConvertTo-Json -Depth 10

Write-Host ""
Write-Host "Recent query logs:"
Get-Content "data\processed\query_logs.jsonl" -Tail 3
