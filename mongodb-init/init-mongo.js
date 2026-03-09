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

// ============================================================================
// ENUMERATION DATABASE SETUP
// ============================================================================

print('\n[1/2] Creating enumeration_db database...');

// Switch to enumeration_db database (creates it if doesn't exist)
db = db.getSiblingDB('enumeration_db');

print('  ✓ Database "enumeration_db" created/selected');

// Create the 'skus' collection
print('\n  Creating "skus" collection...');
db.createCollection('skus', {
    validator: {
        $jsonSchema: {
            bsonType: "object",
            required: ["_id", "tradeNumber", "customerName", "customerType", "productType",
                       "unitsPerCut", "prodPlant", "minWeight", "maxWeight", "targetWeight",
                       "allowedParts"],
            properties: {
                _id: {
                    bsonType: "string",
                    description: "Trade number - must be a unique string (same as tradeNumber)"
                },
                tradeNumber: {
                    bsonType: "string",
                    description: "Trade number identifier"
                },
                customerName: {
                    bsonType: "string",
                    description: "Customer name"
                },
                customerType: {
                    bsonType: "string",
                    description: "Customer type code (e.g., FDS)"
                },
                productType: {
                    bsonType: "string",
                    description: "Product type (e.g., NUGGET, TENDER)"
                },
                unitsPerCut: {
                    bsonType: "int",
                    minimum: 1,
                    description: "Number of units per cut operation"
                },
                prodPlant: {
                    bsonType: "string",
                    description: "Production plant code"
                },
                minWeight: {
                    bsonType: "double",
                    minimum: 0,
                    description: "Minimum weight in grams"
                },
                maxWeight: {
                    bsonType: "double",
                    minimum: 0,
                    description: "Maximum weight in grams"
                },
                targetWeight: {
                    bsonType: "double",
                    minimum: 0,
                    description: "Target weight in grams"
                },
                allowedParts: {
                    bsonType: "array",
                    minItems: 1,
                    items: {
                        bsonType: "string"
                    },
                    description: "Array of allowed part identifiers"
                }
            }
        }
    },
    validationLevel: "moderate",
    validationAction: "warn"
});

print('  ✓ Collection "skus" created with schema validation');

// Create indexes for the skus collection
print('\n  Creating indexes for "skus" collection...');

// Index on customerType for filtering by customer type
db.skus.createIndex(
    { "customerType": 1 },
    {
        name: "idx_customer_type",
        background: true
    }
);
print('    ✓ Index created: idx_customer_type');

// Index on productType for filtering by product type
db.skus.createIndex(
    { "productType": 1 },
    {
        name: "idx_product_type",
        background: true
    }
);
print('    ✓ Index created: idx_product_type');

// Index on prodPlant for filtering by production plant
db.skus.createIndex(
    { "prodPlant": 1 },
    {
        name: "idx_prod_plant",
        background: true
    }
);
print('    ✓ Index created: idx_prod_plant');

// Compound index for common multi-criteria searches
db.skus.createIndex(
    { "customerType": 1, "productType": 1, "prodPlant": 1 },
    {
        name: "idx_customer_product_plant",
        background: true
    }
);
print('    ✓ Index created: idx_customer_product_plant (compound)');

// Index on customerName for text searches
db.skus.createIndex(
    { "customerName": 1 },
    {
        name: "idx_customer_name",
        background: true
    }
);
print('    ✓ Index created: idx_customer_name');

// Create the 'mixes' collection
print('\n  Creating "mixes" collection...');
db.createCollection('mixes', {
    validator: {
        $jsonSchema: {
            bsonType: "object",
            required: ["_id", "skus", "includesFDS", "includesRTL", "includesNug",
                       "numFillets", "filletWeight", "mfgType", "cutStrategyID", "beltSpeed", "skuSetKey"],
            properties: {
                _id: {
                    bsonType: "string",
                    description: "Unique mix ObjectId"
                },
                skus: {
                    bsonType: "object",
                    description: "Map of SKU trade number to Part ID"
                },
                includesFDS: {
                    bsonType: "bool",
                    description: "Whether this mix includes a food service customer"
                },
                includesRTL: {
                    bsonType: "bool",
                    description: "Whether this mix includes a retail customer"
                },
                includesNug: {
                    bsonType: "bool",
                    description: "Whether this mix includes a nugget SKU"
                },
                nuggetTargetWeight: {
                    description: "Target weight of one nugget (nullable, required > 0 when includesNug=true)"
                },
                numFillets: {
                    bsonType: "int",
                    minimum: 0,
                    description: "Count of fillet SKUs in this mix"
                },
                filletWeight: {
                    bsonType: "double",
                    minimum: 0,
                    description: "Total weight of fillet SKUs in this mix"
                },
                mfgType: {
                    enum: ["DSI", "DB20"],
                    description: "Manufacturing line type"
                },
                cutStrategyID: {
                    bsonType: "string",
                    description: "Predetermined cut strategy ID"
                },
                beltSpeed: {
                    bsonType: "double",
                    minimum: 0,
                    description: "Required belt speed for this mix"
                },
                skuSetKey: {
                    bsonType: "string",
                    description: "Deterministic key built from sorted SKU trade numbers for uniqueness enforcement"
                }
            }
        }
    },
    validationLevel: "moderate",
    validationAction: "warn"
});

print('  ✓ Collection "mixes" created with schema validation');

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
// CONFIG DATABASE SETUP
// ============================================================================

print('\n[2/2] Creating config_db database...');

// Switch to config_db database (creates it if doesn't exist)
db = db.getSiblingDB('config_db');

print('  ✓ Database "config_db" created/selected');

// Create the 'global_config' collection
print('\n  Creating "global_config" collection...');
db.createCollection('global_config', {
    validator: {
        $jsonSchema: {
            bsonType: "object",
            required: ["_id", "key", "value", "valueType", "description", "updatedAt"],
            properties: {
                _id: {
                    bsonType: "string",
                    description: "Configuration key - must be a unique string (same as key)"
                },
                key: {
                    bsonType: "string",
                    description: "Configuration key identifier"
                },
                value: {
                    description: "Configuration value (can be int, string, double, or bool)"
                },
                valueType: {
                    enum: ["int", "string", "float", "bool"],
                    description: "Type of the configuration value"
                },
                description: {
                    bsonType: "string",
                    description: "Description of the configuration parameter"
                },
                updatedAt: {
                    bsonType: "date",
                    description: "Last update timestamp"
                },
                minValue: {
                    description: "Minimum allowed value (for numeric types, optional)"
                },
                maxValue: {
                    description: "Maximum allowed value (for numeric types, optional)"
                }
            }
        }
    },
    validationLevel: "moderate",
    validationAction: "warn"
});

print('  ✓ Collection "global_config" created with schema validation');

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

// Insert default configuration values
print('\n  Inserting default configuration values...');
db.global_config.insertMany([
    {
        "_id": "enumeration.defaultMaxTrim",
        "key": "enumeration.defaultMaxTrim",
        "value": 2,
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
        "value": "DSI,DB20",
        "valueType": "string",
        "description": "Comma-separated list of available manufacturing line types for mix selection",
        "updatedAt": new Date()
    }
]);

print('  ✓ Default configuration values inserted');

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

// Verify config_db
db = db.getSiblingDB('config_db');
print('\nconfig_db:');
print('  Collections: ' + db.getCollectionNames().join(', '));
print('  Config indexes: ' + db.global_config.getIndexes().length);
print('  Config documents: ' + db.global_config.countDocuments({}));

print('\n========================================');
print('MongoDB Initialization Complete!');
print('========================================\n');

