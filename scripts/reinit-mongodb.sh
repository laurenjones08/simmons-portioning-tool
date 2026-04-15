#!/usr/bin/env bash
# MongoDB Database Reinitialization Script
#
# This script drops and recreates the MongoDB databases, then re-runs the initialization script.
# If a bootstrap archive exists in mongodb-init/, it restores the exact captured database state instead.
#
# Usage:
#   ./scripts/reinit-mongodb.sh

set -e

echo "=========================================="
echo "MongoDB Database Reinitialization"
echo "=========================================="
echo ""
echo "WARNING: This will DELETE all data in the following databases:"
echo "  - enumeration_db"
echo "  - config_db"
echo ""
read -p "Are you sure you want to continue? (yes/no): " -r
echo ""

if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
    echo "Reinitialization cancelled."
    exit 0
fi

echo "Dropping databases..."

docker exec -it mongodb mongosh -u root -p example --quiet --eval "
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
"

echo ""
echo "Re-applying bootstrap state..."

if [ -f mongodb-init/mongodb-bootstrap.archive.gz ]; then
    echo "Bootstrap archive detected. Restoring exact database state..."
    docker exec mongodb mongorestore -u root -p example --authenticationDatabase admin --drop --gzip --archive=/docker-entrypoint-initdb.d/mongodb-bootstrap.archive.gz
    docker exec -it mongodb mongosh -u root -p example --quiet --eval "db = db.getSiblingDB('local'); db.bootstrap_metadata.updateOne({ _id: 'mongodb-bootstrap-archive-applied' }, { \$set: { appliedAt: new Date(), archive: 'mongodb-bootstrap.archive.gz' } }, { upsert: true });"
else
    echo "No bootstrap archive found. Re-running initialization script..."
    docker exec -it mongodb mongosh -u root -p example --quiet < mongodb-init/init-mongo.js
fi

echo ""
echo "=========================================="
echo "Database reinitialization complete!"
echo "=========================================="
echo ""
echo "Databases recreated:"
echo "  - enumeration_db (with skus and mixes collections)"
echo "  - config_db (with global_config collection)"
echo ""
