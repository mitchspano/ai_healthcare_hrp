# AI Agent Scaffolding for T1D Management

## Project Overview
Build an AI agent to help children with Type 1 diabetes manage their blood sugar using CGM and related data. The agent will provide proactive checks, respond to user queries, and initiate alerts as needed.

---

## Agent Modes

### 1. Conversational Assistant
- Chat interface for children/teens to ask questions about their glucose, insulin, meals, symptoms, etc.
- Example queries:
  - “How is my blood sugar trending?”
  - “Can I eat this candy bar?”
  - “I don’t feel good – am I high or low?”

### 2. Proactive Coach
- Monitors readings and behaviors throughout the day.
- Sends supportive or actionable messages, e.g.:
  - “You were in range all day today, keep it up!”
  - “[12:30] You usually eat around 1pm and take insulin at the same time. If you plan on having lunch again at 1, you can reduce your blood glucose spike if you take your insulin now.”
  - “Great job staying active this afternoon!”
  - “Your blood sugar has been rising overnight. Consider discussing a basal adjustment with your care team.”

---

## Data Model
- **Source:** CGM CSV files (per subject)
- **Fields:**
  - EventDateTime (timestamp)
  - DeviceMode
  - BolusType
  - Basal
  - CorrectionDelivered
  - TotalBolusInsulinDelivered
  - FoodDelivered
  - CarbSize
  - CGM (glucose value)

---

## Core Agent Capabilities
- **Proactive Checks:**
  - Overnight BG rise (3–6am): Suggest basal adjustment if BG rises >30 mg/dL over several nights
  - Weekly status/performance summaries
- **Reactive Q&A:**
  - Trend analysis ("How is my blood sugar trending?")
  - Food/timing advice ("Can I eat this candy bar?", "How long should I wait until I eat this sandwich?")
  - Symptom check ("I don’t feel good – am I high or low?")
- **Agent-Initiated Alerts:**
  - Rising/falling BG
  - Missed bolus
  - Exercise detection
  - Emergency contact notification for critical lows (<50 mg/dL)

---

## Tech Stack (Flexible)
- **Backend:** Python (Flask, FastAPI, or OpenAI Agent SDK)
- **Agent Framework:** LangChain, OpenAI Agent SDK, or similar
- **UI:** Simple web app (minimal effort)
- **Automation:** N8N/MCP server (optional/stretch)

---

## Scaffold Outline
1. **Data Ingestion**
   - Read and parse subject CSVs
   - Normalize and store data in memory or DB
2. **Data Model**
   - Python class or Pydantic model for each row/event
3. **Agent Logic**
   - Proactive checks (overnight, weekly)
   - Q&A and alert logic
4. **API Endpoints**
   - Ingest data
   - Query agent (user questions)
   - Receive agent-initiated alerts
5. **Web UI**
   - User input (text)
   - Agent responses/alerts

---

## Recommended MVP Directory Structure


ai_healthcare_hrp/
│
├── agents.md
├── README.md
├── requirements.txt
│
├── src/                        # Main application package
│   ├── __init__.py
│   ├── main.py                 # Entry point (Flask/FastAPI/OpenAI Agent SDK)
│   ├── config.py               # Configurations and settings
│   │
│   ├── data/                   # Data ingestion and modeling
│   │   ├── __init__.py
│   │   ├── ingestion.py        # CSV reading/parsing
│   │   ├── models.py           # Pydantic/data models
│   │
│   ├── agent/                  # Agent logic
│   │   ├── __init__.py
│   │   ├── proactive.py        # Proactive checks & messages
│   │   ├── conversational.py   # Q&A/chat logic
│   │
│   ├── api/                    # API endpoints
│   │   ├── __init__.py
│   │   ├── routes.py           # REST endpoints for UI/agent
│   │
│   └── utils/                  # Utility functions/helpers
│       ├── __init__.py
│       └── ... 
│
├── ui/                         # Simple web UI (optional for MVP)
│   ├── static/
│   ├── templates/
│   └── app.py                  # UI entry (if separate from backend)
│
└── tests/                      # Unit and integration tests
    ├── __init__.py
    └── ...


- All core logic lives in `app/` (data, agent, API).
- `ui/` is for a minimal web interface (can be Flask templates, React, etc.).
- `tests/` for test coverage.
- `requirements.txt` for dependencies.

This structure is flexible for Flask, FastAPI, or OpenAI Agent SDK, and can be adapted as you grow.

---

## Next Steps
- Implement data ingestion and modeling
- Prototype agent logic for one proactive check and one Q&A
- Scaffold backend with placeholder endpoints
- Build minimal UI for interaction
