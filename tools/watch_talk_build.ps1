param([int]$IntervalSec = 60)
$repo = "cccoll-2026/ai-6657"
$logUrl = "https://cnb.cool/$repo/-/build/logs"
$done = $false
while (-not $done) {
    try {
        $page = Invoke-WebRequest -Uri $logUrl -UseBasicParsing -TimeoutSec 30
        $text = $page.Content
        if ($text -match 'build-talk') {
            if ($text -match 'success|Succeed|passed') {
                Write-Host "[$(Get-Date -Format HH:mm:ss)] ✅ build-talk 构建成功！" -ForegroundColor Green
                $done = $true
            } elseif ($text -match 'fail|error|FATAL|Error') {
                Write-Host "[$(Get-Date -Format HH:mm:ss)] ❌ build-talk 构建失败！" -ForegroundColor Red
                $done = $true
            } else {
                Write-Host "[$(Get-Date -Format HH:mm:ss)] ⏳ 构建中..." -ForegroundColor Yellow
            }
        } else {
            Write-Host "[$(Get-Date -Format HH:mm:ss)] ⏳ 等待构建触发..." -ForegroundColor Gray
        }
    } catch {
        Write-Host "[$(Get-Date -Format HH:mm:ss)] ⚠️ 请求失败: $_" -ForegroundColor DarkYellow
    }
    if (-not $done) { Start-Sleep -Seconds $IntervalSec }
}
