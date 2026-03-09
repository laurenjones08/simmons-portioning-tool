#!/bin/bash
# Script to run the search property tests

echo "Running search property tests..."
pytest services/test_sku_service_properties.py -k "search" -v --tb=short --hypothesis-show-statistics

echo ""
echo "Test run complete!"
