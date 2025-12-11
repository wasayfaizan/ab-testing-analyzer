#!/bin/bash

# Start Backend Server
echo "Starting A/B Testing Analyzer Backend..."
cd backend

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies if needed
if [ ! -f "venv/.installed" ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
    touch venv/.installed
fi

# Start server
echo "Starting FastAPI server on http://localhost:8000"
uvicorn main:app --reload --port 8000

