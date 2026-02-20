#!/bin/bash

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$ROOT_DIR/hragent/bin/python"
STREAMLIT="$ROOT_DIR/hragent/bin/streamlit"

# Renk kodları
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}         HR Agent - Starting Up         ${NC}"
echo -e "${GREEN}========================================${NC}"

# .env kontrolü
if [ ! -f "$ROOT_DIR/.env" ]; then
    echo -e "${RED}ERROR: .env file not found!${NC}"
    echo "Please create a .env file with OPENAI_API_KEY and JWT_SECRET_KEY."
    exit 1
fi

# OPENAI_API_KEY kontrolü
source "$ROOT_DIR/.env" 2>/dev/null || true
if [ -z "$OPENAI_API_KEY" ] || [ "$OPENAI_API_KEY" = "your_openai_api_key_here" ]; then
    echo -e "${RED}ERROR: OPENAI_API_KEY is not set in .env!${NC}"
    echo "Please add your real OpenAI API key to .env"
    exit 1
fi

# hragent env kontrolü
if [ ! -f "$PYTHON" ]; then
    echo -e "${RED}ERROR: hragent virtual environment not found!${NC}"
    echo "Run: python3 -m venv hragent && hragent/bin/pip install -r requirements.txt"
    exit 1
fi

# Backend başlat (arka planda)
echo -e "\n${YELLOW}[1/2] Starting Backend (FastAPI)...${NC}"
cd "$ROOT_DIR/app/backend"
"$PYTHON" agent_backend.py &
BACKEND_PID=$!

# Backend'in ayağa kalkmasını bekle
echo -e "      Waiting for backend to be ready..."
for i in $(seq 1 60); do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo -e "      ${GREEN}Backend is up! (PID: $BACKEND_PID)${NC}"
        break
    fi
    sleep 1
    if [ $i -eq 60 ]; then
        echo -e "      ${RED}Backend did not start in time. Check logs above.${NC}"
        kill $BACKEND_PID 2>/dev/null
        exit 1
    fi
done

# Frontend başlat (ön planda — log buraya gelir)
echo -e "\n${YELLOW}[2/2] Starting Frontend (Streamlit)...${NC}"
echo -e "      ${GREEN}Open http://localhost:8501 in your browser${NC}\n"

cd "$ROOT_DIR/app/frontend"
"$STREAMLIT" run frontend.py --server.port 8501 --server.headless true &
FRONTEND_PID=$!

echo -e "${GREEN}========================================${NC}"
echo -e "  Backend  → http://localhost:8000"
echo -e "  Frontend → http://localhost:8501"
echo -e "  Press Ctrl+C to stop everything."
echo -e "${GREEN}========================================${NC}\n"

# Ctrl+C gelince ikisini de durdur
trap "echo -e '\n${YELLOW}Shutting down...${NC}'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" SIGINT SIGTERM

wait
