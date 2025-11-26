# Nysus Automated Assistant for MES (NAAS)

## Overview

NAAS (Nysus Automated Assistant for MES) is a conversational support assistant built to help MES (Manufacturing Execution System) teams quickly troubleshoot issues, retrieve historical support ticket knowledge, and run live, schema-aware database queries. It combines retrieval-augmented generation (RAG) from a ticket vectorstore with on-demand SQL querying via an MCP (Model Context Protocol) agent and LLM-based synthesis.

This repository hosts the reference implementation used by Nysus engineers to prototype and operate NAAS locally or in a controlled production environment.

## Key Features

- Conversational Streamlit UI with chat history and file upload support
- RAG: similarity search over historical support tickets (Chroma vectorstore)
- MCP Agent: natural-language → SQL generation, schema inspection, and safe execution
- Multi-agent architecture (Scanner, Frontier, Ensemble, Planning, MCP)
- Pluggable LLM backends (OpenAI, Ollama) and embedding providers
- Local memory caching (`memory.json`) for recently asked questions
- Threaded logging and streaming logs to the UI

## Repository Layout

- `nysus_chatbot.py` — Streamlit application (UI, session state, chat flow)
- `chatbot_agent_framework.py` — High-level orchestrator, memory management
- `memory.json` — Local cached questions / memory file
- `agents/` — All agents and data models
  - `agent.py` — Base agent class
  - `planning_agent.py` — Orchestration + LLM synthesis
  - `scanner_agent.py` — Selects cached questions from memory
  - `frontier_agent.py` — RAG retrieval from Chroma vectorstore
  - `ensemble_agent.py` — Aggregates Frontier + MCP results
  - `mcp_agent.py` — Schema inspection, SQL generation, execution
  - `tickets.py` — Typed datamodels for Ticket, Company, etc.
- `data/` — Local data, vectorstores, and knowledge base
  - `documents/os_tickets_vectorstore/frontier` — Chroma persist directory for ticket embeddings
  - `documents/knowledge_base` — Source markdown files used to populate the vectorstore
- `data_parsing_scripts/` — Scripts to initialize / transform data for RAG
- `styles/` — UI styling used by Streamlit

## High-Level Architecture

1. User asks a question in the Streamlit UI.
2. `PlanningAgent` coordinates: `ScannerAgent` (cached questions), `EnsembleAgent`.
3. `EnsembleAgent` runs `FrontierAgent` (RAG search) and `MCPAgent` (database query).
4. `PlanningAgent` synthesizes a natural response using LLM(s), referencing tickets and DB results.
5. UI displays the answer with links to matched tickets and optional database output.

Diagram suggestion for slides: User → Streamlit UI → Planner → {Scanner, Frontier, MCP} → Data stores (Chroma, SQL Server) → Planner → UI

## Quickstart (Development)

Prerequisites

- Python 3.10 or newer
- Git
- A virtual environment tool (recommended: `venv`)
- (Optional) OpenAI API key or Ollama endpoint if using on-prem model
- If you plan to use the MCP Agent against SQL Server: `pyodbc` and network access to the SQL Server instance

Install and run locally (PowerShell commands)

```powershell
# clone (if not already)
git clone <repo-url> "Nysus Chatbot"
cd "Nysus Chatbot"

# create venv and activate
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# install dependencies (see recommended list below)
pip install -r requirements.txt

# run the Streamlit app
streamlit run nysus_chatbot.py
```

If you do not have a `requirements.txt` yet, see the **Dependencies** section below for a recommended starting list.

## Dependencies (recommended)

Create a `requirements.txt` with at least the following packages (versions may vary depending on your environment):

```
streamlit
langchain
openai
chromadb
langchain-openai
langchain-chroma
langchain-ollama
langchain-embeddings
pyodbc
python-dotenv
numpy
scikit-learn
pandas
```

Note: The project uses multiple LangChain connectors and LLM backends. For local testing, you can comment/uncomment specific providers depending on which LLM/embedding you plan to use (OpenAI vs Ollama).

## Configuration

- Environment variables: store OpenAI API key and any other secrets in a `.env` file or your secrets manager.
  - Example `.env` keys: `OPENAI_API_KEY`, `OLLAMA_API_URL`
- Database connection: the Streamlit sidebar allows interactive entry of SQL Server `Host`, `Port`, `User`, and `Password`. Alternatively, set them in `.env` and modify the app to read them directly.

## Vectorstore (RAG) - Rebuild & Initialization

If you need to recreate or update the Chroma vectorstore used by `FrontierAgent`, use the parsing / initialization scripts under `data_parsing_scripts/`.

Example:

```powershell
python data_parsing_scripts/initialize_vectorstore_frontier.py
```

This script reads the markdown knowledge base, computes embeddings (using the configured embedding provider), and writes to `data/documents/os_tickets_vectorstore/frontier`.

## Running with MCP (SQL) Agent

1. Provide database connection parameters in the Streamlit sidebar or via `.env`.
2. Use a test account with least required privileges (read-only is preferred).
3. MCP agent will:
   - Inspect schema (caches schema in memory)
   - Generate SQL from natural language using an LLM
   - Execute vetted queries (SELECT/EXEC/WITH allowed patterns)
   - Return formatted results to the Planner for synthesis

Security Guidance:

- Use service accounts with minimal privileges.
- Mask sensitive fields before returning results to the UI.
- Limit number of rows and columns returned.

## Testing

- Unit tests: Add focused tests for ticket parsing, MPC SQL safety checks, and end-to-end prompt handling.
- Manual testing: Run Streamlit locally and exercise typical user queries, verifying that tickets and DB results are surfaced correctly.

## Operational Considerations

- Authentication: Add RBAC for the Streamlit UI or place NAAS behind an authentication proxy.
- Secrets management: Use a secrets store for DB credentials and API keys.
- Monitoring: Collect metrics for LLM usage, DB queries, and app latency.
- Scaling: For heavy usage, separate the UI from agent execution (e.g., run agents behind a worker queue) and cache frequent LLM results.

## Developer Notes & File Map

- `nysus_chatbot.py`: Streamlit app; session state keys include `agent_framework`, `chat_history`, `matching_tickets`, `matching_databases`, and `db_schema`.
- `chatbot_agent_framework.py`: Reads/writes `memory.json`, initializes `PlanningAgent`, and exposes `run()`.
- `agents/planning_agent.py`: Orchestrates calls to `ScannerAgent` and `EnsembleAgent` and synthesizes LLM responses. Contains the `system_prompt` used to control response style.
- `agents/scanner_agent.py`: Uses an LLM (Ollama/OpenAI) to pick best cached questions from `memory.json`.
- `agents/frontier_agent.py`: Uses embeddings and Chroma to retrieve relevant tickets and parse ticket metadata.
- `agents/ensemble_agent.py`: Coordinates `FrontierAgent` and `MCPAgent` to create a consolidated response object.
- `agents/mcp_agent.py`: Manages DB connections, schema inspection, natural → SQL generation, execution, and result formatting.

## Security & Privacy

- Always run the MCP Agent with least-privileged DB credentials.
- Implement PII redaction for returned ticket content and DB query results.
- Avoid logging raw credentials; mask any secrets in logs.

## Roadmap & Enhancements

- Add PII redaction and automated sanitization of ticket text
- Add RBAC and authentication for the UI
- Extend to multi-tenant vectorstores with tenant-scoped policies
- Integrate monitoring (Prometheus/Grafana) and request tracing
- Add tests for `FrontierAgent` parsing and `MCPAgent` SQL safety

## Contributing

1. Fork the repository
2. Create a feature branch
3. Open a pull request with a clear description and tests

Please follow the repository coding style and run unit tests before submitting a PR.

## License

This project does not include a license in the repository by default. Add a `LICENSE` file (e.g., Apache-2.0 or MIT) to indicate project licensing.

## Contact

For questions about NAAS or this repository, contact the engineering team or open an issue in the repo.

---
