/**
 * MongoDB Initialization Script
 *
 * This script automatically creates the required databases, collections, and indexes
 * for the Simmons Portioning Tool microservices stack.
 *
 * Databases Created:
 * 1. enumeration_db - Stores SKU (Stock Keeping Unit) data
 * 2. config_db - Stores global configuration key-value pairs
 *
 * This script runs automatically when the MongoDB container starts for the first time.
 */

print('========================================');
print('MongoDB Initialization Script Starting');
print('========================================');

const bootstrapMarkerDb = db.getSiblingDB('local');
const bootstrapMarker =
    bootstrapMarkerDb.getCollectionNames().includes('bootstrap_metadata')
        ? bootstrapMarkerDb.bootstrap_metadata.findOne({ "_id": "mongodb-bootstrap-archive-applied" })
        : null;

if (bootstrapMarker) {
    print('MongoDB bootstrap archive already restored; skipping schema/default bootstrap.');
    quit();
}

function ensureCollection(collectionName) {
    if (db.getCollectionNames().includes(collectionName)) {
        print('  Collection "' + collectionName + '" already exists; skipping creation');
        return false;
    }

    db.createCollection(collectionName);
    print('  ✓ Collection "' + collectionName + '" created');
    return true;
}

function ensureDocumentById(collectionName, document) {
    db.getCollection(collectionName).updateOne(
        { "_id": document._id },
        { $setOnInsert: document },
        { upsert: true }
    );
}

// ============================================================================
// ENUMERATION DATABASE SETUP
// ============================================================================

print('\n[1/2] Creating enumeration_db database...');

// Switch to enumeration_db database (creates it if doesn't exist)
db = db.getSiblingDB('enumeration_db');

print('  ✓ Database "enumeration_db" created/selected');

// Create the 'skus' collection
print('\n  Creating "skus" collection...');
ensureCollection('skus');

// Create indexes for the skus collection
print('\n  Creating indexes for "skus" collection...');
db.skus.createIndex({ "customerType": 1 }, { name: "idx_customer_type", background: true });
print('    ✓ Index created: idx_customer_type');
db.skus.createIndex({ "productType": 1 }, { name: "idx_product_type", background: true });
print('    ✓ Index created: idx_product_type');
db.skus.createIndex({ "prodPlant": 1 }, { name: "idx_prod_plant", background: true });
print('    ✓ Index created: idx_prod_plant');
db.skus.createIndex(
    { "customerType": 1, "productType": 1, "prodPlant": 1 },
    { name: "idx_customer_product_plant", background: true }
);
print('    ✓ Index created: idx_customer_product_plant (compound)');
db.skus.createIndex({ "customerName": 1 }, { name: "idx_customer_name", background: true });
print('    ✓ Index created: idx_customer_name');

// Create the 'mixes' collection
print('\n  Creating "mixes" collection...');
ensureCollection('mixes');

// Create indexes for the mixes collection
print('\n  Creating indexes for "mixes" collection...');

// Unique compound index on mfgType + skuSetKey (enforces only one mix per SKU set + mfg type)
db.mixes.createIndex(
    { "mfgType": 1, "skuSetKey": 1 },
    {
        name: "uniq_mfg_type_sku_set_key",
        unique: true,
        background: true
    }
);
print('    ✓ Index created: uniq_mfg_type_sku_set_key (unique compound)');

// Index on mfgType for filtering by manufacturing line
db.mixes.createIndex(
    { "mfgType": 1 },
    {
        name: "idx_mfg_type",
        background: true
    }
);
print('    ✓ Index created: idx_mfg_type');

// Index on includesFDS for filtering by food service mixes
db.mixes.createIndex(
    { "includesFDS": 1 },
    {
        name: "idx_includes_fds",
        background: true
    }
);
print('    ✓ Index created: idx_includes_fds');

// Index on includesRTL for filtering by retail mixes
db.mixes.createIndex(
    { "includesRTL": 1 },
    {
        name: "idx_includes_rtl",
        background: true
    }
);
print('    ✓ Index created: idx_includes_rtl');

// Index on includesNug for filtering by nugget mixes
db.mixes.createIndex(
    { "includesNug": 1 },
    {
        name: "idx_includes_nug",
        background: true
    }
);
print('    ✓ Index created: idx_includes_nug');

// Index on cutStrategyID for filtering by cut strategy
db.mixes.createIndex(
    { "cutStrategyID": 1 },
    {
        name: "idx_cut_strategy_id",
        background: true
    }
);
print('    ✓ Index created: idx_cut_strategy_id');

// Compound index for common search patterns (mfgType + boolean flags)
db.mixes.createIndex(
    { "mfgType": 1, "includesFDS": 1, "includesRTL": 1, "includesNug": 1 },
    {
        name: "idx_mfg_type_flags",
        background: true
    }
);
print('    ✓ Index created: idx_mfg_type_flags (compound)');

// Create collections for staged enumeration runs and results
print('\n  Creating "enumeration_runs" collection...');
ensureCollection('enumeration_runs');

print('\n  Creating "enumeration_results" collection...');
ensureCollection('enumeration_results');

print('\n  Creating "job_status" collection (enumeration-worker-api)...');
ensureCollection('job_status');

print('\n  Creating indexes for enumeration run tracking...');
db.enumeration_runs.createIndex(
    { "status": 1 },
    {
        name: "idx_run_status",
        background: true
    }
);
print('    ✓ Index created: idx_run_status');

db.enumeration_results.createIndex(
    { "runId": 1, "comboKey": 1 },
    {
        name: "uniq_run_combo",
        unique: true,
        background: true
    }
);
print('    ✓ Index created: uniq_run_combo (unique compound)');

db.enumeration_results.createIndex(
    { "runId": 1, "stage": 1 },
    {
        name: "idx_run_stage",
        background: true
    }
);
print('    ✓ Index created: idx_run_stage');

db.enumeration_results.createIndex(
    { "runId": 1, "skuTradeNumbers": 1 },
    {
        name: "idx_run_skus",
        background: true
    }
);
print('    ✓ Index created: idx_run_skus');

db.job_status.createIndex({ "status": 1 }, { name: "idx_job_status_status", background: true });
db.job_status.createIndex({ "runId": 1 }, { name: "idx_job_status_run_id", background: true });
db.job_status.createIndex({ "createdAt": -1 }, { name: "idx_job_status_created_at", background: true });
print('    ✓ Indexes created for "job_status"');

// Insert sample SKU data for testing (optional - uncomment if needed)
/*
print('\n  Inserting sample SKU data...');
db.skus.insertMany([
    {
        "_id": "50624",
        "tradeNumber": "50624",
        "customerName": "CHICK FIL A INC",
        "customerType": "FDS",
        "productType": "NUGGET",
        "unitsPerCut": 8,
        "prodPlant": "FSP",
        "minWeight": 16.0,
        "maxWeight": 19.0,
        "targetWeight": 17.5,
        "allowedParts": ["TENDER", "BREAST"]
    },
    {
        "_id": "50625",
        "tradeNumber": "50625",
        "customerName": "WENDYS",
        "customerType": "FDS",
        "productType": "TENDER",
        "unitsPerCut": 4,
        "prodPlant": "FSP",
        "minWeight": 20.0,
        "maxWeight": 24.0,
        "targetWeight": 22.0,
        "allowedParts": ["TENDER"]
    }
]);
print('  ✓ Sample SKU data inserted');
*/

print('\n✓ Enumeration database setup complete!');

// ============================================================================
// SCHEDULING DATABASE SETUP
// ============================================================================

print('\n[2/3] Creating scheduling_db database...');

db = db.getSiblingDB('scheduling_db');

print('  ✓ Database "scheduling_db" created/selected');

print('\n  Creating "monthly_contracts" collection...');
ensureCollection('monthly_contracts');
print('\n  Creating indexes for "monthly_contracts" collection...');
db.monthly_contracts.createIndex({ "skuId": 1, "yearMonth": 1 }, { name: "uniq_sku_year_month", unique: true, background: true });
db.monthly_contracts.createIndex({ "skuId": 1 }, { name: "idx_monthly_contract_sku_id", background: true });
db.monthly_contracts.createIndex({ "yearMonth": 1 }, { name: "idx_monthly_contract_year_month", background: true });
print('    ✓ Indexes created for "monthly_contracts"');

print('\n  Creating "scheduling_decisions" collection...');
ensureCollection('scheduling_decisions');
print('\n  Creating indexes for "scheduling_decisions" collection...');
db.scheduling_decisions.createIndex({ "mixId": 1, "lineId": 1, "date": 1 }, { name: "uniq_mix_line_date", unique: true, background: true });
db.scheduling_decisions.createIndex({ "mixId": 1 }, { name: "idx_scheduling_decision_mix_id", background: true });
db.scheduling_decisions.createIndex({ "lineId": 1 }, { name: "idx_scheduling_decision_line_id", background: true });
db.scheduling_decisions.createIndex({ "date": 1 }, { name: "idx_scheduling_decision_date", background: true });
print('    ✓ Indexes created for "scheduling_decisions"');

print('\n  Creating "scheduling_outputs" collection...');
ensureCollection('scheduling_outputs');
print('\n  Creating indexes for "scheduling_outputs" collection...');
db.scheduling_outputs.createIndex({ "decisionId": 1, "skuId": 1 }, { name: "uniq_decision_sku", unique: true, background: true });
db.scheduling_outputs.createIndex({ "decisionId": 1 }, { name: "idx_scheduling_output_decision_id", background: true });
db.scheduling_outputs.createIndex({ "skuId": 1 }, { name: "idx_scheduling_output_sku_id", background: true });
db.scheduling_outputs.createIndex({ "date": 1 }, { name: "idx_scheduling_output_date", background: true });
print('    ✓ Indexes created for "scheduling_outputs"');

print('\n  Creating "bucket_usage" collection...');
ensureCollection('bucket_usage');
print('\n  Creating indexes for "bucket_usage" collection...');
db.bucket_usage.createIndex({ "bucketId": 1, "date": 1 }, { name: "uniq_bucket_date", unique: true, background: true });
db.bucket_usage.createIndex({ "bucketId": 1 }, { name: "idx_bucket_usage_bucket_id", background: true });
db.bucket_usage.createIndex({ "date": 1 }, { name: "idx_bucket_usage_date", background: true });
print('\n  Creating "available_wip" collection...');
ensureCollection('available_wip');
print('\n  Creating indexes for "available_wip" collection...');
db.available_wip.createIndex({ "plantName": 1, "bucketId": 1 }, { name: "uniq_plant_bucket", unique: true, background: true });
db.available_wip.createIndex({ "plantName": 1 }, { name: "idx_available_wip_plant_name", background: true });
db.available_wip.createIndex({ "bucketId": 1 }, { name: "idx_available_wip_bucket_id", background: true });
print('    ✓ Indexes created for "available_wip"');
print('    ✓ Indexes created for "bucket_usage"');

print('\n✓ Scheduling database setup complete!');

// ============================================================================
// CONFIG DATABASE SETUP
// ============================================================================

print('\n[3/3] Creating config_db database...');

// Switch to config_db database (creates it if doesn't exist)
db = db.getSiblingDB('config_db');

print('  ✓ Database "config_db" created/selected');

// Create the 'global_config' collection
print('\n  Creating "global_config" collection...');
ensureCollection('global_config');

// Create indexes for the global_config collection
print('\n  Creating indexes for "global_config" collection...');

// Index on valueType for filtering by type
db.global_config.createIndex(
    { "valueType": 1 },
    {
        name: "idx_value_type",
        background: true
    }
);
print('    ✓ Index created: idx_value_type');

// Index on updatedAt for sorting by last update
db.global_config.createIndex(
    { "updatedAt": -1 },
    {
        name: "idx_updated_at",
        background: true
    }
);
print('    ✓ Index created: idx_updated_at');

print('\n  Creating "lines" collection...');
ensureCollection('lines');
db.lines.createIndex({ "lineId": 1 }, { name: "uniq_line_id", unique: true, background: true });
db.lines.createIndex({ "isActive": 1 }, { name: "idx_lines_active", background: true });
print('    ✓ Indexes created for "lines" collection');

// Insert default configuration values
print('\n  Ensuring default configuration values...');
[
    {
        "_id": "enumeration.defaultMaxTrim",
        "key": "enumeration.defaultMaxTrim",
        "value": 15,
        "valueType": "int",
        "description": "Default maximum trim allowed for SKU selection",
        "updatedAt": new Date(),
        "minValue": 0,
        "maxValue": 100
    },
    {
        "_id": "enumeration.defaultMinTargetDelta",
        "key": "enumeration.defaultMinTargetDelta",
        "value": 0.5,
        "valueType": "float",
        "description": "Default minimum target weight delta in grams",
        "updatedAt": new Date(),
        "minValue": 0.0,
        "maxValue": 10.0
    },
    {
        "_id": "enumeration.bucketWeightTolerancePct",
        "key": "enumeration.bucketWeightTolerancePct",
        "value": 0.0,
        "valueType": "float",
        "description": "Tolerance percent applied when fitting mixes to bucket target weight",
        "updatedAt": new Date(),
        "minValue": 0.0,
        "maxValue": 100.0
    },
    {
        "_id": "enumeration.fdsValueCoefficient",
        "key": "enumeration.fdsValueCoefficient",
        "value": 0.0,
        "valueType": "float",
        "description": "Value coefficient applied to total FDS weight during mix scoring",
        "updatedAt": new Date(),
        "minValue": 0.0
    },
    {
        "_id": "enumeration.rtlValueCoefficient",
        "key": "enumeration.rtlValueCoefficient",
        "value": 0.0,
        "valueType": "float",
        "description": "Value coefficient applied to total RTL weight during mix scoring",
        "updatedAt": new Date(),
        "minValue": 0.0
    },
    {
        "_id": "enumeration.trimValueCoefficient",
        "key": "enumeration.trimValueCoefficient",
        "value": 0.0,
        "valueType": "float",
        "description": "Value coefficient applied to trim weight during mix scoring",
        "updatedAt": new Date(),
        "minValue": 0.0
    },
    {
        "_id": "system.enableDebugLogging",
        "key": "system.enableDebugLogging",
        "value": false,
        "valueType": "bool",
        "description": "Enable debug-level logging across all services",
        "updatedAt": new Date()
    },
    {
        "_id": "system.serviceName",
        "key": "system.serviceName",
        "value": "Simmons Portioning Tool",
        "valueType": "string",
        "description": "Name of the application system",
        "updatedAt": new Date()
    },
    {
        "_id": "mix.availablePlants",
        "key": "mix.availablePlants",
        "value": "FSP,SS2,VBS",
        "valueType": "string",
        "description": "Comma-separated list of available production plants for mix selection",
        "updatedAt": new Date()
    },
    {
        "_id": "mix.availableBirdSizes",
        "key": "mix.availableBirdSizes",
        "value": "SB,BB",
        "valueType": "string",
        "description": "Comma-separated list of available bird sizes for mix selection",
        "updatedAt": new Date()
    },
    {
        "_id": "mix.availableMfgTypes",
        "key": "mix.availableMfgTypes",
        "value": "DSI888,DSI884,DB20",
        "valueType": "string",
        "description": "Comma-separated list of available manufacturing line types for mix selection",
        "updatedAt": new Date()
    },
    {
        "_id": "scheduling.gamma_value",
        "key": "scheduling.gamma_value",
        "value": 0.1,
        "valueType": "float",
        "description": "Objective penalty weight for week 1 demand deviation",
        "updatedAt": new Date(),
        "minValue": 0.0,
        "maxValue": 10.0
    }
].forEach(function (document) {
    ensureDocumentById('global_config', document);
});

print('  ✓ Default configuration values ensured');

print('\n✓ Config database setup complete!');

// ============================================================================
// VERIFICATION
// ============================================================================

print('\n========================================');
print('Verification');
print('========================================');

// Verify enumeration_db
db = db.getSiblingDB('enumeration_db');
print('\nenumeration_db:');
print('  Collections: ' + db.getCollectionNames().join(', '));
print('  SKU indexes: ' + db.skus.getIndexes().length);
print('  SKU documents: ' + db.skus.countDocuments({}));
print('  MIX indexes: ' + db.mixes.getIndexes().length);
print('  MIX documents: ' + db.mixes.countDocuments({}));
print('  Enumeration run indexes: ' + db.enumeration_runs.getIndexes().length);
print('  Enumeration run documents: ' + db.enumeration_runs.countDocuments({}));
print('  Enumeration result indexes: ' + db.enumeration_results.getIndexes().length);
print('  Enumeration result documents: ' + db.enumeration_results.countDocuments({}));
print('  Job status indexes: ' + db.job_status.getIndexes().length);
print('  Job status documents: ' + db.job_status.countDocuments({}));

// Verify scheduling_db
db = db.getSiblingDB('scheduling_db');
print('\nscheduling_db:');
print('  Collections: ' + db.getCollectionNames().join(', '));
print('  Scheduling decision indexes: ' + db.scheduling_decisions.getIndexes().length);
print('  Scheduling decision documents: ' + db.scheduling_decisions.countDocuments({}));
print('  Scheduling output indexes: ' + db.scheduling_outputs.getIndexes().length);
print('  Scheduling output documents: ' + db.scheduling_outputs.countDocuments({}));
print('  Monthly contract indexes: ' + db.monthly_contracts.getIndexes().length);
print('  Monthly contract documents: ' + db.monthly_contracts.countDocuments({}));
print('  Bucket usage indexes: ' + db.bucket_usage.getIndexes().length);
print('  Bucket usage documents: ' + db.bucket_usage.countDocuments({}));
print('  Available WIP indexes: ' + db.available_wip.getIndexes().length);
print('  Available WIP documents: ' + db.available_wip.countDocuments({}));

// Verify config_db
db = db.getSiblingDB('config_db');
print('\nconfig_db:');
print('  Collections: ' + db.getCollectionNames().join(', '));
print('  Config indexes: ' + db.global_config.getIndexes().length);
print('  Config documents: ' + db.global_config.countDocuments({}));

print('\n========================================');
print('MongoDB Initialization Complete!');
print('========================================\n');


