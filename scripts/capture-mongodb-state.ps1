param(
    [string]$OutputPath = $(Join-Path (Resolve-Path (Join-Path $PSScriptRoot "..")).Path "mongodb-init\mongodb-bootstrap.archive.gz"),
    [string]$ContainerName = "mongodb"
)

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$resolvedOutputPath = [System.IO.Path]::GetFullPath((Join-Path $repoRoot "mongodb-init\mongodb-bootstrap.archive.gz"))
if ($OutputPath) {
    $resolvedOutputPath = [System.IO.Path]::GetFullPath($OutputPath)
}

$outputDirectory = Split-Path -Parent $resolvedOutputPath
New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null

$containerArchivePath = "/tmp/mongodb-bootstrap.archive.gz"

Write-Host "Capturing MongoDB state from container '$ContainerName'..."
docker exec $ContainerName mongodump -u root -p example --authenticationDatabase admin --archive=$containerArchivePath --gzip

Write-Host "Copying archive into the repository at $resolvedOutputPath..."
docker cp "$($ContainerName):$containerArchivePath" $resolvedOutputPath

Write-Host "Cleaning up temporary archive in the container..."
docker exec $ContainerName rm -f $containerArchivePath

Write-Host "MongoDB state captured successfully."
Write-Host "Archive: $resolvedOutputPath"
