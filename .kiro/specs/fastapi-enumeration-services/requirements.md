# Requirements Document

## Introduction

This document specifies the requirements for a microservices-based enumeration engine system. The system consists of two FastAPI services (Enumeration API and Global Config API) that interact with a MongoDB database. The system provides SKU enumeration capabilities and centralized configuration management, deployed using Docker containers.

## Glossary

- **Enumeration API**: A FastAPI service that manages SKU (Stock Keeping Unit) data and provides search capabilities
- **Global Config API**: A FastAPI service that manages system-wide configuration key-value pairs
- **SKU**: Stock Keeping Unit - a product identifier with associated metadata including customer, product type, and weight specifications
- **Trade Number**: The unique identifier for a SKU, used as the primary key
- **MongoDB**: A NoSQL document database used for persistent storage
- **Docker**: A containerization platform used to package and deploy the services
- **Repository Layer**: The data access layer that interacts directly with the database
- **Service Layer**: The business logic layer that processes data between repositories and routers
- **Router Layer**: The API endpoint layer that handles HTTP requests and responses
- **Dependency Injection**: A design pattern where dependencies are provided to components rather than created internally

## Requirements

### Requirement 1

**User Story:** As a developer, I want to set up a containerized microservices architecture, so that I can deploy and manage multiple services consistently.

#### Acceptance Criteria

1. THE system SHALL provide a docker-compose.yml file that orchestrates three services: mongodb, enumeration-api, and global-config-api
2. WHEN docker-compose is executed, THE system SHALL start all three services with proper networking between them
3. THE system SHALL configure MongoDB with root username and password credentials
4. THE system SHALL persist MongoDB data using a Docker volume
5. THE system SHALL provide environment variables for MongoDB connection configuration in each API service

### Requirement 2

**User Story:** As a developer, I want each API service to have its own Dockerfile and dependencies, so that services can be built and deployed independently.

#### Acceptance Criteria

1. THE Enumeration API SHALL have a dedicated Dockerfile based on Python 3.11
2. THE Global Config API SHALL have a dedicated Dockerfile based on Python 3.11
3. THE Enumeration API SHALL have a requirements.txt file specifying FastAPI, Uvicorn, PyMongo, Pydantic, and python-dotenv
4. THE Global Config API SHALL have a requirements.txt file specifying FastAPI, Uvicorn, PyMongo, Pydantic, and python-dotenv
5. THE system SHALL provide example .env files for each service with MongoDB connection parameters

### Requirement 3

**User Story:** As a developer, I want the Enumeration API to follow a layered architecture, so that the code is maintainable and testable.

#### Acceptance Criteria

1. THE Enumeration API SHALL implement a repository layer for database operations
2. THE Enumeration API SHALL implement a service layer for business logic
3. THE Enumeration API SHALL implement a router layer for HTTP endpoint handling
4. THE Enumeration API SHALL use dependency injection to provide service instances to routers
5. THE Enumeration API SHALL include a config.py module for configuration management
6. THE Enumeration API SHALL include a database.py module for MongoDB connection management

### Requirement 4

**User Story:** As a client application, I want to check the health status of the Enumeration API, so that I can monitor service availability.

#### Acceptance Criteria

1. THE Enumeration API SHALL expose a GET /health endpoint
2. WHEN the /health endpoint is called, THE Enumeration API SHALL return a 200 status code with a success message

### Requirement 5

**User Story:** As a client application, I want to retrieve SKU data by trade number, so that I can access specific product information.

#### Acceptance Criteria

1. THE Enumeration API SHALL expose a GET /skus/{trade_number} endpoint
2. WHEN a valid trade_number is provided, THE Enumeration API SHALL return the corresponding SKU document from the "skus" collection
3. WHEN an invalid trade_number is provided, THE Enumeration API SHALL return a 404 status code
4. THE Enumeration API SHALL use the trade_number as the MongoDB _id field for lookups

### Requirement 6

**User Story:** As a client application, I want to search for SKUs using filter criteria, so that I can find products matching specific attributes.

#### Acceptance Criteria

1. THE Enumeration API SHALL expose a POST /skus/search endpoint
2. WHEN search criteria are provided in the request body, THE Enumeration API SHALL query the "skus" collection with matching filters
3. THE Enumeration API SHALL return an array of SKU documents that match the search criteria
4. WHEN no SKUs match the criteria, THE Enumeration API SHALL return an empty array

### Requirement 7

**User Story:** As a developer, I want SKU documents to follow a consistent schema, so that data integrity is maintained across the system.

#### Acceptance Criteria

1. THE Enumeration API SHALL store SKU documents with _id field equal to the tradeNumber value
2. THE Enumeration API SHALL store SKU documents with the following required fields: tradeNumber, customerName, customerType, productType, unitsPerCut, prodPlant, minWeight, maxWeight, targetWeight, birdSize, allowedParts
3. THE Enumeration API SHALL validate that minWeight, maxWeight, and targetWeight are numeric values
4. THE Enumeration API SHALL validate that allowedParts is an array of strings
5. THE Enumeration API SHALL use Pydantic models for SKU schema validation

### Requirement 8

**User Story:** As a developer, I want the Global Config API to follow a layered architecture, so that configuration management is maintainable.

#### Acceptance Criteria

1. THE Global Config API SHALL implement a repository layer for database operations
2. THE Global Config API SHALL implement a service layer for business logic
3. THE Global Config API SHALL implement a router layer for HTTP endpoint handling
4. THE Global Config API SHALL use dependency injection to provide service instances to routers
5. THE Global Config API SHALL include a config.py module for configuration management
6. THE Global Config API SHALL include a database.py module for MongoDB connection management

### Requirement 9

**User Story:** As a client application, I want to check the health status of the Global Config API, so that I can monitor service availability.

#### Acceptance Criteria

1. THE Global Config API SHALL expose a GET /health endpoint
2. WHEN the /health endpoint is called, THE Global Config API SHALL return a 200 status code with a success message

### Requirement 10

**User Story:** As a system administrator, I want to retrieve configuration values by key, so that I can view current system settings.

#### Acceptance Criteria

1. THE Global Config API SHALL expose a GET /config/{key} endpoint
2. WHEN a valid configuration key is provided, THE Global Config API SHALL return the configuration document from the "global_config" collection
3. WHEN an invalid key is provided, THE Global Config API SHALL return a 404 status code
4. THE Global Config API SHALL use the key as the MongoDB _id field for lookups

### Requirement 11

**User Story:** As a system administrator, I want to update configuration values, so that I can modify system behavior without redeploying services.

#### Acceptance Criteria

1. THE Global Config API SHALL expose a PUT /config/{key} endpoint
2. WHEN a configuration update is submitted, THE Global Config API SHALL update the document in the "global_config" collection
3. WHEN updating a configuration, THE Global Config API SHALL set the updatedAt field to the current timestamp
4. WHEN a new configuration key is provided, THE Global Config API SHALL create a new document in the collection
5. THE Global Config API SHALL validate that the value matches the specified valueType (int, string, float, bool)

### Requirement 12

**User Story:** As a system administrator, I want to retrieve all configuration values, so that I can review the complete system configuration.

#### Acceptance Criteria

1. THE Global Config API SHALL expose a GET /config endpoint
2. WHEN the /config endpoint is called, THE Global Config API SHALL return all documents from the "global_config" collection
3. THE Global Config API SHALL return configuration documents as an array

### Requirement 13

**User Story:** As a developer, I want configuration documents to follow a consistent schema, so that configuration data is structured and validated.

#### Acceptance Criteria

1. THE Global Config API SHALL store configuration documents with _id field equal to the key value
2. THE Global Config API SHALL store configuration documents with the following required fields: key, value, valueType, description, updatedAt
3. THE Global Config API SHALL validate that valueType is one of: "int", "string", "float", "bool"
4. THE Global Config API SHALL use Pydantic models for configuration schema validation
5. THE Global Config API SHALL store updatedAt as an ISO 8601 formatted datetime string

### Requirement 14

**User Story:** As a developer, I want comprehensive code comments and documentation, so that the codebase is easy to understand and maintain.

#### Acceptance Criteria

1. THE system SHALL include comments in all Dockerfile files explaining each instruction
2. THE system SHALL include comments in the docker-compose.yml file explaining service configurations
3. THE Enumeration API SHALL include docstrings for all functions and classes
4. THE Global Config API SHALL include docstrings for all functions and classes
5. THE system SHALL include inline comments explaining complex logic or business rules
