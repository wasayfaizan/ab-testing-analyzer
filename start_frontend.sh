#!/bin/bash

# Start Frontend Server
echo "Starting A/B Testing Analyzer Frontend..."
cd frontend

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "Installing dependencies..."
    npm install
fi

# Start development server
echo "Starting Vite dev server on http://localhost:5173"
npm run dev

