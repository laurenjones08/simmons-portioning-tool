#!/usr/bin/env bash
# MongoDB Database Reinitialization Script
#
# This script drops and recreates the MongoDB databases, then re-runs the initialization script.
# Use this to reset your development database to a clean state.
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

# Run commands inside the MongoDB container
docker exec -it mongodb mongosh -u root -p example --quiet --eval "
print('Dropping enumeration_db...');
db.getSiblingDB('enumeration_db').dropDatabase();
print('✓ enumeration_db dropped');

print('Dropping config_db...');
db.getSiblingDB('config_db').dropDatabase();
print('✓ config_db dropped');
"

echo ""
echo "Re-running initialization script..."

# Re-run the init script
docker exec -it mongodb mongosh -u root -p example --quiet < mongodb-init/init-mongo.js

echo ""
echo "=========================================="
echo "Database reinitialization complete!"
echo "=========================================="
echo ""
echo "Databases recreated:"
echo "  - enumeration_db (with skus and mixes collections)"
echo "  - config_db (with global_config collection)"
echo ""

