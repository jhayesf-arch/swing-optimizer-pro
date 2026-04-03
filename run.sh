#!/bin/bash
cd "$(dirname "$0")/backend" || exit 1
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

echo "=================================================="
echo "    SWING OPTIMIZER PRO SERVER STARTING...        "
echo "    Access the dashboard at: http://localhost:8000"
echo "=================================================="

uvicorn main:app --host 0.0.0.0 --port 8000 --reload
