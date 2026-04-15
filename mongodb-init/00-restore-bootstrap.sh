#!/bin/sh
set -eu

BOOTSTRAP_ARCHIVE="/docker-entrypoint-initdb.d/mongodb-bootstrap.archive.gz"

echo "Checking for MongoDB bootstrap archive..."

if [ -f "$BOOTSTRAP_ARCHIVE" ]; then
  echo "Bootstrap archive found. Restoring database state from $BOOTSTRAP_ARCHIVE"
  mongorestore \
    --username "$MONGO_INITDB_ROOT_USERNAME" \
    --password "$MONGO_INITDB_ROOT_PASSWORD" \
    --authenticationDatabase admin \
    --drop \
    --gzip \
    --archive="$BOOTSTRAP_ARCHIVE"

  mongosh \
    --username "$MONGO_INITDB_ROOT_USERNAME" \
    --password "$MONGO_INITDB_ROOT_PASSWORD" \
    --authenticationDatabase admin \
    --quiet \
    --eval "db = db.getSiblingDB('local'); db.bootstrap_metadata.updateOne({ _id: 'mongodb-bootstrap-archive-applied' }, { \$set: { appliedAt: new Date(), archive: 'mongodb-bootstrap.archive.gz' } }, { upsert: true });"

  echo "MongoDB bootstrap archive restored."
else
  echo "No bootstrap archive found. Skipping restore and continuing with schema bootstrap."
fi
