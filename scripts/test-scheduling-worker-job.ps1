param(
    [string]$BaseUrl = $env:SCHEDULING_WORKER_API_URL,
    [string]$EnumerationApiUrl = $env:ENUMERATION_API_URL,
    [string]$PlantId = "VBS",
    [string[]]$SkuIds = @(),
    [string]$RunId = "",
    [string]$OutputDir = "outputs",
    [int]$HorizonDays = 12,
    [int]$RandomSkuCount = 15,
    [string]$PlanStartDate = (Get-Date).ToString("yyyy-MM-dd"),
    [switch]$SaveCsv,
    [switch]$Tee,
    [string]$ShortTermFile = ""
)

$ErrorActionPreference = "Stop"

function Write-Info([string]$Message) {
    Write-Host "[info] $Message"
}

function Format-JobProgress {
    param(
        [Parameter(Mandatory = $true)]$Job
    )

    $state = [string]$Job.status
    $stage = [string]$Job.currentStage
    $message = [string]$Job.stageMessage
    $elapsed = $null
    if ($Job.timings -and $Job.timings.job_elapsed -ne $null) {
        $elapsed = [double]$Job.timings.job_elapsed
    }

    $parts = @($state)
    if (-not [string]::IsNullOrWhiteSpace($stage)) {
        $parts += "stage=$stage"
    }
    if ($elapsed -ne $null) {
        $parts += ("elapsed={0:n1}s" -f $elapsed)
    }
    if (-not [string]::IsNullOrWhiteSpace($message)) {
        $parts += $message
    }

    return ($parts -join " | ")
}

function Invoke-JsonRequest {
    param(
        [Parameter(Mandatory = $true)][string]$Method,
        [Parameter(Mandatory = $true)][string]$Url,
        [string]$Body = ""
    )

    $headers = @{ Accept = "application/json" }
    if ($Method -eq "GET") {
        return Invoke-RestMethod -Method Get -Uri $Url -Headers $headers
    }

    return Invoke-RestMethod -Method $Method -Uri $Url -Headers $headers -ContentType "application/json" -Body $Body
}

function Get-RandomSkuIdsForPlant {
    param(
        [Parameter(Mandatory = $true)][string]$ApiUrl,
        [Parameter(Mandatory = $true)][string]$TargetPlantId,
        [int]$Count = 15
    )

    $skus = Invoke-JsonRequest -Method "POST" -Url "$ApiUrl/skus/search" -Body "{}"
    if (-not $skus) {
        throw "No SKUs were returned from the enumeration API."
    }

    $skuIds = @(
        $skus |
            Where-Object { $_.tradeNumber -and ($_.prodPlant -eq $TargetPlantId) } |
            ForEach-Object { [string]$_.tradeNumber } |
            Sort-Object -Unique
    )

    if (-not $skuIds) {
        throw "No SKUs matched plant $TargetPlantId after filtering the full SKU catalog."
    }

    if ($skuIds.Count -lt $Count) {
        Write-Info "Only $($skuIds.Count) SKUs matched plant $TargetPlantId; using all available matches."
        return $skuIds
    }

    return $skuIds | Get-Random -Count $Count
}

if ([string]::IsNullOrWhiteSpace($BaseUrl)) {
    $BaseUrl = "http://localhost:8080/api/scheduling-worker"
}

if ([string]::IsNullOrWhiteSpace($EnumerationApiUrl)) {
    $EnumerationApiUrl = "http://localhost:8080/api/enumeration"
}

if ([string]::IsNullOrWhiteSpace($RunId)) {
    $RunId = "schedule-test-{0}" -f (Get-Date).ToString("yyyyMMdd-HHmmss")
}

if (-not $SkuIds -or $SkuIds.Count -eq 0) {
    Write-Info "No SKU IDs provided; selecting $RandomSkuCount random SKUs for plant $PlantId from $EnumerationApiUrl"
    $SkuIds = Get-RandomSkuIdsForPlant -ApiUrl $EnumerationApiUrl -TargetPlantId $PlantId -Count $RandomSkuCount
}

$payload = [ordered]@{
    runId = $RunId
    plantId = $PlantId
    skuIds = @($SkuIds)
    shortTermFile = $(if ([string]::IsNullOrWhiteSpace($ShortTermFile)) { $null } else { $ShortTermFile })
    saveCsv = [bool]$SaveCsv.IsPresent
    outputDir = $OutputDir
    tee = [bool]$Tee.IsPresent
    planStartDate = $PlanStartDate
    horizonDays = $HorizonDays
}

$submitUrl = "$BaseUrl/jobs"
Write-Info "Submitting sample scheduling job to $submitUrl"
Write-Info ("Payload: " + ($payload | ConvertTo-Json -Depth 10 -Compress))

$submitResult = Invoke-JsonRequest -Method "POST" -Url $submitUrl -Body ($payload | ConvertTo-Json -Depth 10)
Write-Host ""
Write-Host "Submitted job:"
$submitResult | ConvertTo-Json -Depth 20

$jobId = $submitResult.jobId
if ([string]::IsNullOrWhiteSpace($jobId)) {
    throw "The worker did not return a jobId."
}

Write-Info "Polling job status for jobId $jobId"
$terminalStates = @("completed", "failed", "cancelled")
do {
    Start-Sleep -Seconds 3
    $status = Invoke-JsonRequest -Method "GET" -Url "$BaseUrl/jobs/$jobId"
    $state = [string]$status.status
    Write-Host ("[{0}] {1}" -f (Get-Date).ToString("HH:mm:ss"), (Format-JobProgress -Job $status))
} while ($terminalStates -notcontains $state)

Write-Host ""
Write-Host "Final job record:"
$status | ConvertTo-Json -Depth 30

if ($state -eq "failed") {
    Write-Host ""
    Write-Host "Failure message:"
    Write-Host $status.errorMessage
    if ($status.errorTraceback) {
        Write-Host ""
        Write-Host "Traceback preview:"
        $trace = [string]$status.errorTraceback
        if ($trace.Length -gt 4000) {
            $trace = $trace.Substring(0, 4000) + "`n`n[traceback truncated]"
        }
        Write-Host $trace
    }
    exit 1
}

Write-Host ""
Write-Host "Job finished successfully."
