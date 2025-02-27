#!/bin/bash

# Run tests with coverage reports
python -m pytest --cov=quick_wins/marketing_chatbot --cov-report=term --cov-report=html:coverage_reports tests/

# Display summary
echo "Test coverage report generated in coverage_reports/"
echo "Open coverage_reports/index.html in your browser to view detailed coverage" 