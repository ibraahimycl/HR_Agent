# HR Agent — AI-Powered HR Assistant

An HR assistant built with a real **Agentic RAG** architecture. Employees and HR staff can ask natural-language questions about company policies, leave balances, and salary — and the agent reasons step-by-step, calling the right tools to answer.

## 🎥 Demo Video

[![Watch the demo](https://img.youtube.com/vi/1SfOI0SOCnxGpvWbhgjzAcCuHRXN3Ruk5/0.jpg)](https://drive.google.com/file/d/1SfOI0SOCnxGpvWbhgjzAcCuHRXN3Ruk5/view?usp=drive_link)

> **Click the image above to watch the demo video on Google Drive.**

---

## How It Works

### Agent Architecture

The system uses **OpenAI Function Calling** (`create_openai_tools_agent`) — the LLM decides which tool to invoke based on the user's question. Every decision is logged and shown in the UI as "Agent Reasoning".



<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/72cdb8c0-c32c-441f-af4d-155fb390612a" />



### RAG Pipeline

Policy documents are indexed into **ChromaDB** using section-aware chunking — each HR policy section (A. Vacation Leave, B. Sick Leave, VI. Leave Encashment, etc.) becomes its own chunk. Section context is injected directly into the text before embedding so the model always knows which section a rule belongs to.

At query time:
1. The query is translated to English (policy is in English)
2. **MMR retrieval** (k=6, fetch_k=25) fetches diverse, relevant chunks
3. The LLM answers strictly from retrieved context and cites sources

### Memory

Each user session maintains:
- **Last 3 messages** verbatim (for immediate context — "bunun encashment'ı?" resolves to the last mentioned employee)
- **Rolling summary** updated every 10 messages (prevents token bloat)
- **`last_mentioned_employees`** list injected into every system prompt (HR can ask about multiple people across turns)

### Security (Role-Based Access Control)

Access control is **deterministic** — enforced in backend code before any tool runs, never left to the LLM.

| Role | Access |
|---|---|
| **Employee** | Own data only. Any query involving another employee's name is blocked at tool level. |
| **HR** | Full access to all employee data. Can query and calculate for multiple employees. |

---

## Tools

| Tool | Description |
|---|---|
| `get_employee_info` | Reads employee data from CSV: salary, leave balances, position, hire date, status |
| `calculate_hr_metrics` | HR calculations: leave encashment, salary raise, benefits cost, overtime pay. Supports multi-employee aggregates. |
| `query_policy` | Semantic search over HR policy documents via ChromaDB MMR. Returns answer + citations. |

---

## Project Structure

```
HR_Agent/
├── app/
│   ├── backend/
│   │   ├── hr_agent.py          # Agent core: tools, memory, RBAC, session state
│   │   ├── agent_backend.py     # FastAPI: auth (JWT), /chat, /token, /logout
│   │   └── store_embeddings.py  # ChromaDB ingestion: section-aware chunking + metadata injection
│   ├── frontend/
│   │   └── frontend.py          # Streamlit UI: chat + Agent Reasoning expander + Policy Sources
│   └── policies/
│       └── hr_policy.txt        # HR policy document (Leave + Attendance)
├── data/
│   └── sample_employee.csv      # Employee database (30 employees)
├── start.sh                     # Single command to start both backend and frontend
├── requirements.txt
└── .env.example
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | OpenAI `gpt-4o-mini` |
| Agent Framework | LangChain `create_openai_tools_agent` |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector Store | ChromaDB (persistent, section-aware) |
| Retrieval | MMR — Maximal Marginal Relevance |
| Backend API | FastAPI + JWT authentication |
| Frontend | Streamlit |
| Data | Pandas + CSV |

---

## Setup

### Prerequisites
- Python 3.9+
- An OpenAI API key

### Installation

```bash
# Clone the repository
git clone https://github.com/ibraahimycl/HR_Agent.git
cd HR_Agent

# Create virtual environment
python3 -m venv hragent
source hragent/bin/activate  # Windows: hragent\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY and JWT_SECRET_KEY
```

### Generate a JWT Secret Key

```bash
python3 -c "import secrets; print(secrets.token_hex(32))"
```

### Run

```bash
bash start.sh
```

The script starts both backend and frontend, waits for the backend to be ready, then prints:

```
Backend  → http://localhost:8000
Frontend → http://localhost:8501
```

On first run, ChromaDB index is created automatically (takes ~20-30 seconds). Subsequent starts are instant.

Press `Ctrl+C` to stop everything.

---

## Example Queries

**As an Employee:**
- "How many days of leave do I have left?"
- "What is my salary?"
- "What are the leave encashment rules?"

**As HR:**
- "What is Emily Baker's encashment amount?"
- "Tell me about Aaron Myers." → "What is this person's salary?"
- "What is the total encashment cost for Emily Baker, Aaron Myers, and Jordan Frazier?"
- "Which leave types cannot be used by probationary employees?"

---

## Environment Variables

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | Your OpenAI API key |
| `JWT_SECRET_KEY` | Secret string for JWT signing (any random hex) |
| `LLM_MODEL` | LLM model name (default: `gpt-4o-mini`) |
| `API_URL` | Backend URL for frontend (default: `http://localhost:8000`) |

---

## License

Developed for educational and portfolio demonstration purposes.
