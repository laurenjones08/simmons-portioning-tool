param(
    [string]$BaseUrl = $env:SCHEDULING_WORKER_API_URL,
    [string]$SchedulingApiUrl = $env:SCHEDULING_API_URL,
    [string]$EnumerationApiUrl = $env:ENUMERATION_API_URL,
    [string]$PlantId = "VBS",
    [string[]]$SkuIds = @(),
    [string]$RunId = "",
    [string]$OutputDir = "outputs",
    [int]$HorizonDays = 12,
    [int]$RandomSkuCount = 15,
    [int]$RequestTimeoutSec = 0,
    [string]$PlanStartDate = (Get-Date).ToString("yyyy-MM-dd"),
    [switch]$SaveCsv,
    [switch]$Tee,
    [switch]$DebugMode,
    [string]$ShortTermFile = ""
)

$ErrorActionPreference = "Stop"
$script:TranscriptStarted = $false
$script:ExitCode = 0
$script:StagedShortTermHostPath = $null

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
        if ($RequestTimeoutSec -le 0) {
            return Invoke-RestMethod -Method Get -Uri $Url -Headers $headers -TimeoutSec 0
        }
        return Invoke-RestMethod -Method Get -Uri $Url -Headers $headers -TimeoutSec $RequestTimeoutSec
    }

    if ($RequestTimeoutSec -le 0) {
        return Invoke-RestMethod -Method $Method -Uri $Url -Headers $headers -ContentType "application/json" -Body $Body -TimeoutSec 0
    }
    return Invoke-RestMethod -Method $Method -Uri $Url -Headers $headers -ContentType "application/json" -Body $Body -TimeoutSec $RequestTimeoutSec
}

function Save-DebugDataPrepDump {
    param(
        [Parameter(Mandatory = $true)][string]$ApiBaseUrl,
        [Parameter(Mandatory = $true)][string]$JobId,
        [Parameter(Mandatory = $true)][string]$RunLabel
    )

    $debugUrl = "$ApiBaseUrl/jobs/$JobId/debug-data-prep"
    $debugPath = Join-Path $script:ResolvedOutputDir "$RunLabel-debug-data-prep-$JobId.json"
    try {
        Invoke-WebRequest -Method Get `
        -Uri $debugUrl `
        -OutFile $debugPath
        Write-Info "Saved debug dataprep dump to $debugPath"

    } catch {
        Write-Info "No debug dataprep dump was available at $debugUrl"
        return
    }
}

function Save-JobArtifacts {
    param(
        [Parameter(Mandatory = $true)][string]$ApiBaseUrl,
        [Parameter(Mandatory = $true)][string]$JobId,
        [Parameter(Mandatory = $true)][string]$RunLabel
    )

    $artifacts = Invoke-JsonRequest -Method "GET" -Url "$ApiBaseUrl/jobs/$JobId/artifacts"
    if (-not $artifacts) {
        Write-Info "No CSV artifacts were returned for job $JobId"
        return
    }

    foreach ($artifact in @($artifacts)) {
        $artifactName = [string]$artifact.artifactName
        $fileName = [string]$artifact.fileName
        $downloadUrl = [string]$artifact.downloadUrl
        if ([string]::IsNullOrWhiteSpace($downloadUrl)) {
            Write-Info "Skipping artifact $artifactName because no downloadUrl was returned"
            continue
        }

        if ([string]::IsNullOrWhiteSpace($fileName)) {
            $fileName = "$artifactName.csv"
        }

        $artifactPath = Join-Path $script:ResolvedOutputDir "$RunLabel-$fileName"
        Invoke-WebRequest -Method Get -Uri $downloadUrl -OutFile $artifactPath
        Write-Info "Saved CSV artifact to $artifactPath"
    }
}

function Resolve-InputPath {
    param(
        [Parameter(Mandatory = $true)][string]$PathValue
    )

    if ([string]::IsNullOrWhiteSpace($PathValue)) {
        return ""
    }

    if ([System.IO.Path]::IsPathRooted($PathValue)) {
        return [System.IO.Path]::GetFullPath($PathValue)
    }

    $candidateFromCurrent = Join-Path (Get-Location) $PathValue
    if (Test-Path -LiteralPath $candidateFromCurrent) {
        return [System.IO.Path]::GetFullPath($candidateFromCurrent)
    }

    return [System.IO.Path]::GetFullPath((Join-Path $repoRoot $PathValue))
}

function Stage-ShortTermDemandFile {
    param(
        [Parameter(Mandatory = $true)][string]$SourcePath,
        [Parameter(Mandatory = $true)][string]$RunLabel
    )

    $resolvedSourcePath = Resolve-InputPath -PathValue $SourcePath
    if (-not (Test-Path -LiteralPath $resolvedSourcePath)) {
        throw "Short-term demand file not found: $resolvedSourcePath"
    }

    $rows = Import-Csv -LiteralPath $resolvedSourcePath
    if (-not $rows) {
        throw "Short-term demand file was empty: $resolvedSourcePath"
    }

    $normalizedRows = foreach ($row in $rows) {
        $sku = [string]$row.sku
        $date = [string]$row.date
        $qty = $row.qty

        if ([string]::IsNullOrWhiteSpace($sku) -or [string]::IsNullOrWhiteSpace($date) -or [string]::IsNullOrWhiteSpace([string]$qty)) {
            continue
        }

        $parsedDate = [datetime]::Parse($date)
        $parsedQty = [double]$qty

        [pscustomobject]@{
            sku     = $sku.Trim()
            dueDate = $parsedDate.ToString("yyyy-MM-dd")
            demand  = $parsedQty
            type    = "Short"
        }
    }

    if (-not $normalizedRows) {
        throw "Short-term demand file did not contain any valid rows after reading sku, date, and qty columns."
    }

    $stageDir = Join-Path $repoRoot "scheduling-worker-api\\.short-term-inputs"
    New-Item -ItemType Directory -Path $stageDir -Force | Out-Null
    $stagedFileName = "$RunLabel-short-term-demand.csv"
    $stagedHostPath = Join-Path $stageDir $stagedFileName
    $normalizedRows | Export-Csv -LiteralPath $stagedHostPath -NoTypeInformation

    return @{
        HostPath = $stagedHostPath
        ContainerPath = "/app/.short-term-inputs/$stagedFileName"
        RowCount = @($normalizedRows).Count
    }
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

if ([string]::IsNullOrWhiteSpace($SchedulingApiUrl)) {
    if ($BaseUrl -match "/api/scheduling-worker/?$") {
        $SchedulingApiUrl = ($BaseUrl -replace "/api/scheduling-worker/?$", "/api/scheduling")
    } else {
        $SchedulingApiUrl = "http://localhost:8080/api/scheduling"
    }
}

if ([string]::IsNullOrWhiteSpace($EnumerationApiUrl)) {
    $EnumerationApiUrl = "http://localhost:8080/api/enumeration"
}

if ([string]::IsNullOrWhiteSpace($RunId)) {
    $RunId = "schedule-test-{0}" -f (Get-Date).ToString("yyyyMMdd-HHmmss")
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$script:ResolvedOutputDir = if ([System.IO.Path]::IsPathRooted($OutputDir)) {
    $OutputDir
} else {
    Join-Path $repoRoot $OutputDir
}
New-Item -ItemType Directory -Path $script:ResolvedOutputDir -Force | Out-Null
$logPath = Join-Path $script:ResolvedOutputDir "$RunId-script-output.log"
Start-Transcript -Path $logPath -Force | Out-Null
$script:TranscriptStarted = $true
Write-Info "Capturing script output to $logPath"

$resolvedShortTermFile = $null
if (-not [string]::IsNullOrWhiteSpace($ShortTermFile)) {
    $stagedShortTerm = Stage-ShortTermDemandFile -SourcePath $ShortTermFile -RunLabel $RunId
    $resolvedShortTermFile = [string]$stagedShortTerm.ContainerPath
    $script:StagedShortTermHostPath = [string]$stagedShortTerm.HostPath
    Write-Info "Staged $($stagedShortTerm.RowCount) short-term demand rows to $($stagedShortTerm.HostPath)"
    Write-Info "Worker will read short-term demand file from $resolvedShortTermFile"
}

if (-not $SkuIds -or $SkuIds.Count -eq 0) {
    Write-Info "No SKU IDs provided; selecting $RandomSkuCount random SKUs for plant $PlantId from $EnumerationApiUrl"
    $SkuIds = Get-RandomSkuIdsForPlant -ApiUrl $EnumerationApiUrl -TargetPlantId $PlantId -Count $RandomSkuCount
}

$payload = [ordered]@{
    runId = $RunId
    plantId = $PlantId
    skuIds = @($SkuIds)
    shortTermFile = $resolvedShortTermFile
    saveCsv = [bool]$SaveCsv.IsPresent
    outputDir = $OutputDir
    tee = [bool]$Tee.IsPresent
    debugMode = [bool]$DebugMode.IsPresent
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

if ($DebugMode.IsPresent) {
    Save-DebugDataPrepDump -ApiBaseUrl $BaseUrl -JobId $jobId -RunLabel $RunId
}

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
    $script:ExitCode = 1
} else {
    Write-Host ""
    Write-Host "Job finished successfully."
    Save-JobArtifacts -ApiBaseUrl $SchedulingApiUrl -JobId $jobId -RunLabel $RunId
}

if ($script:StagedShortTermHostPath -and (Test-Path -LiteralPath $script:StagedShortTermHostPath)) {
    Remove-Item -LiteralPath $script:StagedShortTermHostPath -Force
    Write-Info "Deleted staged short-term demand file $script:StagedShortTermHostPath"
}

if ($script:TranscriptStarted) {
    Stop-Transcript | Out-Null
    $script:TranscriptStarted = $false
}

if ($script:ExitCode -ne 0) {
    exit $script:ExitCode
}
