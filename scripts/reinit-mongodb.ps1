# MongoDB Database Reinitialization Script (PowerShell)
#
# This script drops and recreates the MongoDB databases, then re-runs the initialization script.
# If a bootstrap archive exists in mongodb-init/, it restores the exact captured database state instead.
#
# Usage:
#   .\scripts\reinit-mongodb.ps1

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "MongoDB Database Reinitialization" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "WARNING: This will DELETE all data in the following databases:" -ForegroundColor Yellow
Write-Host "  - enumeration_db" -ForegroundColor Yellow
Write-Host "  - config_db" -ForegroundColor Yellow
Write-Host ""

$confirmation = Read-Host "Are you sure you want to continue? (yes/no)"

if ($confirmation -ne "yes") {
    Write-Host "Reinitialization cancelled." -ForegroundColor Red
    exit 0
}

Write-Host ""
Write-Host "Dropping databases..." -ForegroundColor Yellow

$dropScript = @"
print('Dropping enumeration_db...');
db.getSiblingDB('enumeration_db').dropDatabase();
print('✓ enumeration_db dropped');

print('Dropping config_db...');
db.getSiblingDB('config_db').dropDatabase();
print('✓ config_db dropped');

print('Clearing bootstrap marker...');
var localDb = db.getSiblingDB('local');
if (localDb.getCollectionNames().includes('bootstrap_metadata')) {
    localDb.bootstrap_metadata.drop();
    print('✓ bootstrap_metadata cleared');
} else {
    print('✓ bootstrap_metadata not present');
}
"@

docker exec -i mongodb mongosh -u root -p example --quiet --eval $dropScript

Write-Host ""
Write-Host "Re-applying bootstrap state..." -ForegroundColor Yellow

$bootstrapArchive = Join-Path (Resolve-Path (Join-Path $PSScriptRoot "..")).Path "mongodb-init\mongodb-bootstrap.archive.gz"

if (Test-Path $bootstrapArchive) {
    Write-Host "Bootstrap archive detected. Restoring exact database state..." -ForegroundColor Yellow
    docker exec mongodb mongorestore -u root -p example --authenticationDatabase admin --drop --gzip --archive=/docker-entrypoint-initdb.d/mongodb-bootstrap.archive.gz
    $restoreMarkerScript = @'
db = db.getSiblingDB('local');
db.bootstrap_metadata.updateOne({ _id: 'mongodb-bootstrap-archive-applied' }, { $set: { appliedAt: new Date(), archive: 'mongodb-bootstrap.archive.gz' } }, { upsert: true });
'@
    docker exec -i mongodb mongosh -u root -p example --quiet --eval $restoreMarkerScript
} else {
    Write-Host "No bootstrap archive found. Re-running initialization script..." -ForegroundColor Yellow
    Get-Content mongodb-init/init-mongo.js | docker exec -i mongodb mongosh -u root -p example --quiet
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "Database reinitialization complete!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Databases recreated:" -ForegroundColor Cyan
Write-Host "  - enumeration_db (with skus and mixes collections)" -ForegroundColor Cyan
Write-Host "  - config_db (with global_config collection)" -ForegroundColor Cyan
Write-Host ""
