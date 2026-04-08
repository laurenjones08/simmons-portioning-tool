# MongoDB Database Reinitialization Script (PowerShell)
#
# This script drops and recreates the MongoDB databases, then re-runs the initialization script.
# Use this to reset your development database to a clean state.
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

# Drop databases
$dropScript = @"
print('Dropping enumeration_db...');
db.getSiblingDB('enumeration_db').dropDatabase();
print('✓ enumeration_db dropped');

print('Dropping config_db...');
db.getSiblingDB('config_db').dropDatabase();
print('✓ config_db dropped');
"@

docker exec -i mongodb mongosh -u root -p example --quiet --eval $dropScript

Write-Host ""
Write-Host "Re-running initialization script..." -ForegroundColor Yellow

# Re-run the init script
Get-Content mongodb-init/init-mongo.js | docker exec -i mongodb mongosh -u root -p example --quiet

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "Database reinitialization complete!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Databases recreated:" -ForegroundColor Cyan
Write-Host "  - enumeration_db (with skus and mixes collections)" -ForegroundColor Cyan
Write-Host "  - config_db (with global_config collection)" -ForegroundColor Cyan
Write-Host ""

