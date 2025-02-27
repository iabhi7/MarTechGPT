#!/bin/bash

# Netcore AI Marketing Suite Demo Script
# This script launches all components for demonstration purposes

# Set environment variables
export MODEL_NAME="distilgpt2"
export QUANTIZE="True"
export PORT=5000

# Color formatting
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=======================================================${NC}"
echo -e "${BLUE}     Netcore AI Marketing Suite - Demo Launcher        ${NC}"
echo -e "${BLUE}=======================================================${NC}"
echo

# Check dependencies
echo -e "${BLUE}Checking dependencies...${NC}"
dependencies=("python3" "pip" "streamlit" "uvicorn")
for dep in "${dependencies[@]}"; do
    if ! command -v $dep &> /dev/null; then
        echo -e "${RED}Error: $dep is not installed. Please install it first.${NC}"
        exit 1
    fi
done
echo -e "${GREEN}All dependencies found.${NC}"
echo

# Install required packages if not already installed
echo -e "${BLUE}Checking required packages...${NC}"
pip install -r requirements.txt > /dev/null
echo -e "${GREEN}Package check complete.${NC}"
echo

# Function to run a component in the background
run_component() {
    local component=$1
    local command=$2
    echo -e "${BLUE}Starting $component...${NC}"
    $command &
    local pid=$!
    echo -e "${GREEN}$component started with PID $pid${NC}"
    echo "$pid" > "pid_$component.txt"
}

# Start the API server
run_component "API" "uvicorn quick_wins.api_docs.api:app --host 0.0.0.0 --port 5000"

# Start the Streamlit dashboard
run_component "Dashboard" "streamlit run quick_wins/dashboard/app.py --server.port 8501"

echo
echo -e "${BLUE}=======================================================${NC}"
echo -e "${GREEN}All components started successfully!${NC}"
echo -e "${BLUE}API Documentation:${NC} http://localhost:5000/api/docs"
echo -e "${BLUE}Interactive Dashboard:${NC} http://localhost:8501"
echo -e "${BLUE}=======================================================${NC}"
echo
echo "Press Ctrl+C to stop all components"

# Wait for user input
read -p "Press Enter to stop all components..."

# Clean up
echo -e "${BLUE}Stopping all components...${NC}"
if [ -f "pid_API.txt" ]; then
    kill $(cat pid_API.txt)
    rm pid_API.txt
fi

if [ -f "pid_Dashboard.txt" ]; then
    kill $(cat pid_Dashboard.txt)
    rm pid_Dashboard.txt
fi

echo -e "${GREEN}Demo stopped. Thank you for using Netcore AI Marketing Suite!${NC}" 