# Nysus Automated Assistant for MES (NAAS)

## Overview

NAAS (Nysus Automated Assistant for MES) is an intelligent conversational support assistant built to help MES (Manufacturing Execution System) teams quickly troubleshoot issues, retrieve historical support ticket knowledge, and run live, schema-aware database queries. It combines retrieval-augmented generation (RAG) from a ticket vectorstore with on-demand SQL querying via an MCP (Model Context Protocol) agent and LLM-based synthesis.

This repository hosts the reference implementation used by Nysus engineers to prototype and operate NAAS locally or in a controlled production environment.

## Key Features

- **Conversational Streamlit UI** with chat history and PDF file upload support
- **Google OAuth Authentication** with domain-restricted access (@nysus.net, @nysus.com)
- **RAG (Retrieval-Augmented Generation)**: Similarity search over historical support tickets using Chroma vectorstore
- **MCP Agent**: Natural-language → SQL generation, schema inspection, and safe execution across multiple databases
- **Multi-agent Architecture**: Scanner, Frontier, Ensemble, Planning, and MCP agents working in concert
- **Hybrid LLM Support**: Flexible configuration using OpenAI (GPT-4o-mini) and Ollama (Llama3.2) models
- **PDF Document Processing**: Upload and extract context from PDF files to enhance responses
- **Local Memory Caching** (`memory.json`) for recently asked questions with intelligent retrieval
- **Real-time Logging**: Threaded logging with color-coded agent messages displayed in the UI
- **Dynamic Database Connection**: Connect/disconnect from SQL Server with schema caching for performance
- **Dual Sidebar Layout**: Left sidebar for settings/connection, right sidebar for matching tickets and database info

## Repository Layout

```
nysus_chatbot/
├── nysus_chatbot.py                    # Main Streamlit application (UI, session state, chat flow)
├── chatbot_agent_framework.py          # High-level orchestrator, memory management
├── memory.json                         # Local cached questions/memory file
├── auth.py                             # Google OAuth authentication module
├── auth_config.py                      # Authentication configuration and settings
├── log_utils.py                        # Logging utilities and color formatting
├── .env                                # Environment variables (not in repo - see .env.example)
├── agents/                             # All agents and data models
│   ├── agent.py                        # Base agent class with logging
│   ├── base_classes.py                 # Pydantic models (Ticket, Company, Question, etc.)
│   ├── planning_agent.py               # Orchestration + LLM synthesis (OpenAI GPT-4o-mini)
│   ├── scanner_agent.py                # Selects cached questions from memory (Ollama Llama3.2)
│   ├── frontier_agent.py               # RAG retrieval from Chroma vectorstore (OpenAI embeddings + GPT-4o-mini)
│   ├── ensemble_agent.py               # Aggregates Frontier + MCP results
│   ├── mcp_agent.py                    # Schema inspection, SQL generation, execution (OpenAI GPT-4o-mini)
│   ├── open_source_agent.py            # Alternative RAG agent with Ollama embeddings
│   └── specialist_agent.py             # Fine-tuned model agent (Modal integration)
├── data/                               # Local data, vectorstores, and knowledge base
│   ├── documents/
│   │   ├── knowledge_base/             # Source markdown files for RAG
│   │   │   ├── facilities_directory_rag.md
│   │   │   ├── ost_export_*.md         # Support ticket exports by year
│   │   │   ├── projects_index_rag.md
│   │   │   └── quotes_index_rag.md
│   │   ├── os_tickets_vectorstore/     # Chroma persist directories
│   │   │   ├── frontier/               # OpenAI embeddings vectorstore
│   │   │   └── open_source/            # Ollama embeddings vectorstore
│   │   └── misc/                       # CSV data and project/quote RAG files
│   ├── documents_backup/               # Backup of original data
│   └── image/                          # UI assets (logo, backgrounds)
├── data_parsing_scripts/               # Scripts to initialize/transform data for RAG
│   ├── initialize_vectorstore_frontier.py     # Create vectorstore with OpenAI embeddings
│   ├── initialize_vectorstore_open_source.py  # Create vectorstore with Ollama embeddings
│   ├── transform_csv_for_rag.py               # Transform CSV files to RAG format
│   └── transform_tickets_for_rag.py           # Transform ticket exports to RAG format
└── styles/
    └── styles.css                      # Custom UI styling
```

## High-Level Architecture

### Workflow

1. **User Authentication**: Google OAuth validates user email against allowed domains (@nysus.net, @nysus.com)
2. **User Input**: User asks a question in the Streamlit UI (optionally uploads PDF files)
3. **Planning Agent** coordinates the workflow:
   - **Scanner Agent**: Searches cached questions from `memory.json` using Ollama
   - **Ensemble Agent**: Runs in parallel:
     - **Frontier Agent**: RAG search against Chroma vectorstore (OpenAI embeddings)
     - **MCP Agent**: Generates and executes SQL queries across multiple databases
4. **Response Synthesis**: Planning Agent uses OpenAI GPT-4o-mini to create a natural language response combining:
   - Relevant support tickets with similarity scores
   - Database query results with SQL explanations
   - Recently asked similar questions
   - Content from uploaded PDF documents
5. **UI Display**: Response is streamed to the UI with:
   - Main chat area showing synthesized answer
   - Right sidebar displaying matching tickets (sorted by relevance)
   - Right sidebar showing queried databases and tables

### Agent Details

| Agent | Purpose | LLM/Model | Key Features |
|-------|---------|-----------|--------------|
| **Planning Agent** | Orchestrates workflow, synthesizes responses | OpenAI GPT-4o-mini | Coordinates all agents, generates natural responses |
| **Scanner Agent** | Retrieves cached questions | Ollama Llama3.2 | JSON-formatted responses, selects top 5 matches |
| **Frontier Agent** | RAG retrieval from tickets | OpenAI embeddings + GPT-4o-mini | Similarity search, ticket parsing, PDF processing |
| **Ensemble Agent** | Aggregates results | N/A (coordinator) | Runs Frontier + MCP in parallel |
| **MCP Agent** | SQL query generation/execution | OpenAI GPT-4o-mini | Multi-database support, schema caching, safe query execution |
| **Open Source Agent** | Alternative RAG (unused) | Ollama embeddings | Fully local, no API costs |

### Data Flow Diagram

```
┌─────────────┐
│   User      │
│ (Authenticated)
└──────┬──────┘
       │ Question + PDF files
       ↓
┌─────────────────────────────────────────────┐
│        Streamlit UI (nysus_chatbot.py)      │
│  - Session state management                 │
│  - Chat history display                     │
│  - Database connection controls             │
│  - PDF file upload handling                 │
└──────┬──────────────────────────────────────┘
       │
       ↓
┌─────────────────────────────────────────────┐
│  ChatbotAgentFramework                      │
│  - Memory management (memory.json)          │
│  - DB connection params                     │
│  - Agent initialization                     │
└──────┬──────────────────────────────────────┘
       │
       ↓
┌─────────────────────────────────────────────┐
│         Planning Agent (Orchestrator)       │
│  - Coordinates Scanner + Ensemble           │
│  - Synthesizes natural responses            │
└──────┬──────────────────────────────────────┘
       │
       ├────────────────┬────────────────────┐
       ↓                ↓                    ↓
┌─────────────┐  ┌─────────────────┐  ┌──────────────┐
│   Scanner   │  │    Ensemble     │  │   Response   │
│    Agent    │  │     Agent       │  │  Synthesis   │
│             │  │                 │  │              │
│ Searches    │  │  Coordinates:   │  │ Combines all │
│ memory.json │  │  - Frontier     │  │ results with │
│ using       │  │  - MCP          │  │ GPT-4o-mini  │
│ Ollama      │  │                 │  │              │
└─────────────┘  └────┬───────┬────┘  └──────────────┘
                      │       │
            ┌─────────┘       └─────────┐
            ↓                           ↓
    ┌──────────────┐            ┌──────────────┐
    │   Frontier   │            │  MCP Agent   │
    │    Agent     │            │              │
    │              │            │ - Schema     │
    │ - RAG search │            │   inspection │
    │ - Chroma DB  │            │ - SQL gen    │
    │ - OpenAI     │            │ - Multi-DB   │
    │   embeddings │            │   execution  │
    │ - PDF parse  │            │              │
    └──────┬───────┘            └──────┬───────┘
           │                           │
           ↓                           ↓
    ┌──────────────┐            ┌──────────────┐
    │ Chroma       │            │ SQL Server   │
    │ Vectorstore  │            │ (Multiple    │
    │ (Tickets)    │            │  Databases)  │
    └──────────────┘            └──────────────┘
```

## Quickstart (Development)

### Prerequisites

- Python 3.10 or newer
- Git
- Windows PowerShell (primary development environment)
- OpenAI API key (for embeddings and GPT-4o-mini)
- (Optional) Ollama installed locally for Scanner Agent and alternative embeddings
- SQL Server access with ODBC Driver 17 for SQL Server

### Installation & Setup

```powershell
# Clone repository
git clone <repo-url> nysus_chatbot
cd nysus_chatbot

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install core dependencies
pip install streamlit langchain openai chromadb langchain-openai langchain-chroma langchain-ollama pyodbc python-dotenv numpy scikit-learn pandas PyPDF2

# Install authentication dependencies
pip install google-auth-oauthlib google-auth google-api-python-client

# Create .env file from example (if available)
# Copy-Item .env.example .env

# Edit .env and add your credentials
# See "Configuration" section below
```

### Configuration

Create a `.env` file in the project root with the following variables:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Google OAuth Configuration (see Google Authentication Setup below)
GOOGLE_CLIENT_ID=your_google_client_id.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=your_google_client_secret
REDIRECT_URI=http://localhost:8501

# Optional: Ollama Configuration (if using local models)
OLLAMA_API_URL=http://localhost:11434
```

### Running the Application

```powershell
# Ensure virtual environment is activated
.\.venv\Scripts\Activate.ps1

# Run Streamlit app
streamlit run nysus_chatbot.py
```

The application will:
1. Open in your browser at `http://localhost:8501`
2. Show Google OAuth login page
3. Authenticate users with @nysus.net or @nysus.com emails
4. Display the chat interface with database connection controls

## Dependencies

### Core Dependencies

```
streamlit>=1.28.0
langchain>=0.1.0
langchain-openai>=0.0.5
langchain-chroma>=0.1.0
langchain-ollama>=0.0.1
langchain-community>=0.0.20
openai>=1.10.0
chromadb>=0.4.22
pyodbc>=4.0.39
python-dotenv>=1.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
pandas>=2.0.0
PyPDF2>=3.0.0
```

### Authentication Dependencies

```
google-auth-oauthlib>=1.1.0
google-auth>=2.23.0
google-api-python-client>=2.108.0
```

### Optional Dependencies

```
# For local LLM support (Ollama)
ollama>=0.1.0

# For fine-tuned models (Modal)
modal>=0.55.0
```

### Creating requirements.txt

Create a `requirements.txt` file with all dependencies:

```powershell
pip freeze > requirements.txt
```

Or install from the dependencies listed above:

```powershell
pip install streamlit langchain langchain-openai langchain-chroma langchain-ollama langchain-community openai chromadb pyodbc python-dotenv numpy scikit-learn pandas PyPDF2 google-auth-oauthlib google-auth google-api-python-client
```

## LLM Configuration and Cost Optimization

### Current LLM Usage

The application uses a **hybrid approach** combining OpenAI and Ollama models:

| Component | Model | Purpose | Cost |
|-----------|-------|---------|------|
| **Planning Agent** | OpenAI GPT-4o-mini | Response synthesis | Paid API |
| **Scanner Agent** | Ollama Llama3.2 | Cache question selection | Free (local) |
| **Frontier Agent (Embeddings)** | OpenAI text-embedding-3-large | Vector search | Paid API (minimal) |
| **Frontier Agent (LLM)** | OpenAI GPT-4o-mini | Ticket parsing | Paid API |
| **MCP Agent** | OpenAI GPT-4o-mini | SQL generation | Paid API |

### Switching Between OpenAI and Ollama

The codebase is designed for easy switching between providers:

**Scanner Agent** (already using Ollama):
```python
# agents/scanner_agent.py
from langchain_ollama import ChatOllama
self.llm = ChatOllama(model="llama3.2", temperature=0, format="json")
```

**To switch Frontier/MCP to Ollama** (reduce costs):
```python
# agents/frontier_agent.py or mcp_agent.py
# Comment out:
# from openai import OpenAI
# self.client = OpenAI()

# Add:
from langchain_ollama import ChatOllama
self.llm = ChatOllama(model="llama3.2")
```

**Important**: If switching embeddings (OpenAI ↔ Ollama), you must recreate the vectorstore:
```powershell
python data_parsing_scripts/initialize_vectorstore_frontier.py
```

### Database Connection Management

Database connections are managed through the Streamlit UI sidebar:

1. **Interactive Connection**: Enter credentials in the sidebar
   - Host: SQL Server host (supports named pipes: `np:\\.\pipe\LOCALDB#...`)
   - Port: Default 1433
   - User: SQL authentication username
   - Password: SQL authentication password

2. **Connection Features**:
   - Real-time connection testing
   - Schema caching for performance
   - Multi-database support (queries all non-system databases)
   - Connect/Disconnect buttons
   - Persistent connection status indicator (🟢/🔴)

3. **Programmatic Configuration** (optional):
   - Store credentials in `.env` if needed
   - Modify `nysus_chatbot.py` to pre-fill connection fields

## Vectorstore (RAG) - Rebuild & Initialization

The application uses Chroma vectorstores for ticket similarity search. There are two vectorstore configurations:

### Frontier Vectorstore (Default - OpenAI Embeddings)

Located at: `data/documents/os_tickets_vectorstore/frontier`

**To rebuild:**
```powershell
python data_parsing_scripts/initialize_vectorstore_frontier.py
```

This script:
- Reads markdown files from `data/documents/knowledge_base/`
- Creates embeddings using OpenAI `text-embedding-3-large` (dimension 1536)
- Stores in Chroma vectorstore with metadata
- Used by **Frontier Agent**

### Open Source Vectorstore (Alternative - Ollama Embeddings)

Located at: `data/documents/os_tickets_vectorstore/open_source`

**To rebuild:**
```powershell
python data_parsing_scripts/initialize_vectorstore_open_source.py
```

This script:
- Uses Ollama `nomic-embed-text` embeddings (dimension 768, free/local)
- Used by **Open Source Agent** (not currently active in main workflow)
- Useful for fully local, cost-free operation

### Data Transformation Scripts

**Transform ticket exports to RAG format:**
```powershell
python data_parsing_scripts/transform_tickets_for_rag.py
```

**Transform CSV data to RAG format:**
```powershell
python data_parsing_scripts/transform_csv_for_rag.py
```

### Knowledge Base Structure

Source files in `data/documents/knowledge_base/`:
- `ost_export_2018_rag.md` through `ost_export_2025_part_2_rag.md` - Support ticket exports
- `facilities_directory_rag.md` - Facility information
- `projects_index_rag.md` - Project references
- `quotes_index_rag.md` - Quote references

## Running with MCP (SQL) Agent

### Connection Process

1. **Launch Application**: Start Streamlit as described in Quickstart
2. **Authenticate**: Log in with Google OAuth (@nysus.net or @nysus.com)
3. **Configure Database**: In left sidebar, enter SQL Server credentials
4. **Connect**: Click "Connect to SQL Server" button
5. **Schema Caching**: System automatically fetches and caches database schema

### MCP Agent Capabilities

The MCP Agent provides intelligent database querying:

- **Multi-Database Support**: Queries across all non-system databases
- **Schema Inspection**: Automatically discovers tables, columns, relationships
- **Natural Language to SQL**: Converts user questions to valid SQL queries
- **Safe Execution**: Validates queries before execution (SELECT/EXEC/WITH patterns)
- **Result Formatting**: Returns human-readable results with explanations
- **Schema Caching**: Stores schema in session state for performance

### Example Queries

- "Show me all projects for Honda"
- "What are the recent tickets related to barcode scanning?"
- "Find all quotes from 2024 with status pending"
- "Which facilities have the most support tickets?"

### Security Features

✓ **Query Validation**: Only allows safe query patterns (SELECT, EXEC, WITH)
✓ **Least Privilege**: Recommended to use read-only database accounts
✓ **Connection Isolation**: Each session maintains separate credentials
✓ **Schema-Aware**: Uses table/column metadata to generate accurate queries
✓ **Error Handling**: Graceful failure with informative error messages

### Recommended Security Configuration

```python
# Use a dedicated read-only account for NAAS
# Grant permissions:
# - SELECT on user databases
# - EXECUTE on specific stored procedures (if needed)
# - NO write permissions (INSERT/UPDATE/DELETE)
```

### Troubleshooting

**Connection fails:**
- Verify ODBC Driver 17 is installed: `Get-OdbcDriver` in PowerShell
- Check SQL Server network access and firewall rules
- For named pipes: Ensure format is correct (`np:\\.\pipe\...`)
- Test credentials with SQL Server Management Studio first

**Schema not caching:**
- Check user has permissions to query `sys.databases` and `INFORMATION_SCHEMA`
- Verify at least one non-system database exists
- Review error messages in terminal output

**Query generation issues:**
- Ensure schema is cached (indicator shows 🟢)
- Try more specific questions with table/column names
- Check terminal logs for SQL generation errors

## PDF Document Upload and Processing

### Feature Overview

Users can upload PDF documents to provide additional context for their questions. The Frontier Agent extracts text from PDFs and uses it alongside ticket knowledge.

### Usage

1. **Upload**: Use the file uploader in the left sidebar
2. **Multiple Files**: Select multiple PDFs if needed
3. **Ask Questions**: Reference content from uploaded documents in your queries
4. **Context Integration**: Responses automatically incorporate relevant PDF content

### Technical Details

- **Extraction**: PyPDF2 library extracts text from all pages
- **Processing**: Text is chunked using LangChain `CharacterTextSplitter`
- **Embeddings**: Creates in-memory FAISS vectorstore with OpenAI embeddings
- **Retrieval**: Similarity search finds relevant document sections
- **Integration**: Planning Agent synthesizes PDF content with tickets and database results

### Example Workflow

```python
# User uploads project_specification.pdf
# User asks: "What are the testing requirements?"
# System:
# 1. Extracts text from PDF
# 2. Creates temporary FAISS vectorstore
# 3. Searches for "testing requirements"
# 4. Combines with ticket knowledge
# 5. Returns comprehensive answer
```

### Limitations

- PDF text extraction may vary with document format
- Image-based PDFs (scanned) require OCR (not currently supported)
- Large PDFs may take time to process
- Context limited to most relevant sections (not entire document)

### Supported Formats

- ✅ Text-based PDFs
- ✅ Multi-page documents
- ❌ Image-only PDFs (scanned documents without text layer)
- ❌ Password-protected PDFs

## Memory and Caching System

### Memory.json Structure

The application maintains a local cache of recently asked questions in `memory.json`:

```json
[
  {
    "question": "Detailed question description",
    "answer": "Complete resolution",
    "database": {
      "database_name": "table_name"
    },
    "resolution": true
  }
]
```

### Scanner Agent Behavior

The **Scanner Agent** uses Ollama Llama3.2 to:
1. Read all cached questions from `memory.json`
2. Compare user query with cached questions using LLM
3. Select top 5 most relevant cached questions
4. Return structured JSON response with:
   - Detailed question summaries
   - Clear resolutions
   - Relevant database references
   - Resolution status

### Memory Management

**Reading Memory:**
```python
# chatbot_agent_framework.py
def read_memory(self) -> List[Ticket]:
    # Loads from memory.json if exists
    # Returns list of Question objects
```

**Writing Memory:**
```python
def write_memory(self, memory: List[Ticket]):
    # Saves to memory.json
    # Preserves recently asked questions
```

### Benefits

- **Fast Responses**: Instant retrieval of similar past questions
- **Learning System**: Improves with each interaction
- **Cost Reduction**: Uses local Ollama instead of paid API
- **Context Awareness**: References previous solutions

### Memory File Location

- **Development**: `memory.json` in project root
- **Production**: Consider database or cloud storage for persistence

## Testing and Quality Assurance

### Manual Testing

**Basic Functionality:**
```powershell
# 1. Launch application
streamlit run nysus_chatbot.py

# 2. Test authentication flow
# - Verify Google OAuth login
# - Test domain restriction (@nysus.net, @nysus.com)
# - Verify unauthorized domains are rejected

# 3. Test ticket retrieval
# - Ask: "How do I troubleshoot barcode scanner issues?"
# - Verify relevant tickets appear in right sidebar
# - Check similarity scores are displayed

# 4. Test database connectivity
# - Enter SQL Server credentials
# - Click "Connect to SQL Server"
# - Verify schema caching success message
# - Ask: "Show me all projects from 2024"
# - Verify SQL query and results are displayed

# 5. Test PDF upload
# - Upload a technical document
# - Ask questions about its content
# - Verify response incorporates PDF context

# 6. Test cached questions
# - Ask the same question twice
# - Second time should be faster (from memory)
```

### Automated Testing (Recommended)

**Unit Tests** (to be implemented):
```python
# tests/test_ticket_parsing.py
def test_ticket_extraction():
    # Test Frontier Agent ticket parsing
    pass

# tests/test_sql_safety.py
def test_sql_injection_prevention():
    # Test MCP Agent query validation
    pass

# tests/test_authentication.py
def test_domain_restriction():
    # Test auth.py domain validation
    pass
```

**Integration Tests:**
```python
# tests/test_agent_workflow.py
def test_end_to_end_query():
    # Test full workflow from question to response
    pass
```

### Test Coverage Areas

1. **Authentication Module** (`auth.py`)
   - Google OAuth flow
   - Domain validation
   - Session management

2. **Frontier Agent** (`agents/frontier_agent.py`)
   - Ticket parsing from markdown
   - Similarity search accuracy
   - PDF text extraction

3. **MCP Agent** (`agents/mcp_agent.py`)
   - SQL injection prevention
   - Query pattern validation
   - Schema inspection
   - Multi-database handling

4. **Planning Agent** (`agents/planning_agent.py`)
   - Response synthesis quality
   - Context integration
   - Error handling

5. **UI Components** (`nysus_chatbot.py`)
   - Session state management
   - Connection handling
   - File upload processing

### Performance Testing

**Metrics to Monitor:**
- Response time (target: <10s for typical queries)
- OpenAI API costs per query
- Memory usage with large chat histories
- Database query execution time
- PDF processing time

### Quality Checklist

- [ ] All agents log their actions properly
- [ ] Error messages are user-friendly
- [ ] SQL queries are validated before execution
- [ ] Sensitive data is not logged
- [ ] PDF uploads handle malformed files gracefully
- [ ] Database disconnection cleans up resources
- [ ] Session state is properly managed
- [ ] Authentication enforces domain restrictions

## Operational Considerations

### Authentication & Authorization

- **Google OAuth**: Built-in domain restriction for @nysus.net and @nysus.com
- **Session Management**: 24-hour session timeout (configurable in `auth_config.py`)
- **Future Enhancements**: Consider adding role-based access control (RBAC)

### Secrets Management

**Current Implementation:**
- `.env` file for development (not committed to repo)
- Environment variables for OpenAI API key and Google OAuth credentials

**Production Recommendations:**
- Use cloud secrets manager (AWS Secrets Manager, Azure Key Vault, GCP Secret Manager)
- Rotate database credentials regularly
- Never log or display API keys/secrets

### Monitoring & Observability

**Recommended Metrics:**
- LLM API call count and costs (OpenAI)
- Database query execution time
- Agent response time breakdown
- User session duration
- Authentication success/failure rates
- Error rates by component

**Logging:**
- All agents log to console with color-coded output
- Logs displayed in real-time in Streamlit UI
- Consider structured logging (JSON) for production

**Future Integrations:**
- Prometheus for metrics collection
- Grafana for visualization
- ELK Stack or Splunk for log aggregation
- Application Performance Monitoring (APM) tools

### Scaling Considerations

**Current Architecture:**
- Single Streamlit instance
- Synchronous agent execution
- In-memory session state

**Scaling Strategies:**

1. **Horizontal Scaling:**
   - Deploy multiple Streamlit instances behind load balancer
   - Use external session store (Redis) instead of in-memory
   - Share `memory.json` via database or cloud storage

2. **Agent Optimization:**
   - Cache frequent LLM responses
   - Pre-compute embeddings for common queries
   - Implement request queuing for background processing
   - Consider async agent execution

3. **Database Optimization:**
   - Connection pooling for SQL Server
   - Read replicas for query distribution
   - Query result caching
   - Implement rate limiting per user

4. **Cost Optimization:**
   - Switch more components to Ollama (local LLM)
   - Implement intelligent caching to reduce API calls
   - Monitor and optimize prompt sizes
   - Use cheaper models for non-critical tasks

### Deployment Options

**Local/Development:**
```powershell
streamlit run nysus_chatbot.py
```

**Streamlit Cloud:**
- Push to GitHub
- Connect Streamlit Cloud to repository
- Configure secrets in Streamlit Cloud dashboard
- Update `REDIRECT_URI` for OAuth

**Docker (Recommended for Production):**
```dockerfile
# Dockerfile example
FROM python:3.10
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "nysus_chatbot.py"]
```

**Cloud Platforms:**
- AWS: EC2, ECS, or App Runner
- Azure: App Service or Container Instances
- GCP: Cloud Run or Compute Engine

### Backup & Recovery

**Data to Backup:**
- `memory.json` - Question cache
- `.env` - Configuration (securely)
- `data/documents/os_tickets_vectorstore/` - Vectorstore databases
- `data/documents/knowledge_base/` - Source documents

**Backup Strategy:**
```powershell
# Automated backup script
$date = Get-Date -Format "yyyyMMdd"
Compress-Archive -Path "memory.json","data/documents" -DestinationPath "backup_$date.zip"
```

**Recovery:**
- Restore files to their original locations
- Rebuild vectorstores if corrupted: `python data_parsing_scripts/initialize_vectorstore_frontier.py`

## Developer Notes & File Map

### Key Session State Variables

**In `nysus_chatbot.py`:**
- `agent_framework` - Instance of ChatbotAgentFramework
- `chat_history` - List of AIMessage/HumanMessage objects
- `matching_tickets` - List of Ticket objects from current query
- `matching_databases` - List of database/table dicts queried
- `db_schema` - Cached database schema string
- `db_connection_params` - SQL Server connection credentials
- `user_uploaded_files` - List of uploaded PDF files
- `log_data` - Real-time logging messages
- `background_path` - Selected background image

### Agent Workflow

```python
# chatbot_agent_framework.py
ChatbotAgentFramework.run(user_query, chat_history, uploaded_files, schema)
    ↓
PlanningAgent.run(user_query, chat_history, uploaded_files, schema)
    ↓
    ├─→ ScannerAgent.scan(memory, user_query, uploaded_files)
    │   └─→ Returns recently asked similar questions
    │
    ├─→ EnsembleAgent.answer_question(user_query, chat_history, n_results, schema, uploaded_files)
    │   ├─→ FrontierAgent.find_relevant_tickets(user_query, n_results)
    │   │   └─→ Chroma similarity search → List[Ticket]
    │   │
    │   ├─→ FrontierAgent.find_relevant_uploaded_content(user_query, uploaded_files)
    │   │   └─→ PDF extraction → FAISS search → relevant text
    │   │
    │   └─→ MCPAgent.answer_question_with_database(user_query, chat_history, schema)
    │       └─→ SQL generation → execution → results
    │
    └─→ PlanningAgent.synthesize_response(...)
        └─→ GPT-4o-mini combines all sources → natural language response
```

### Important Code Patterns

**Agent Initialization:**
```python
# Lazy initialization pattern
def init_agents_as_needed(self):
    if not self.planner:
        self.planner = PlanningAgent(collection=self.collection, db_connection_params=self.db_connection_params)
```

**Database Connection Update:**
```python
# Update existing agents when connection changes
def update_db_connection_params(self, db_connection_params=None):
    if self.planner and hasattr(self.planner, 'ensemble'):
        self.planner.ensemble.mcp.set_connection_params(...)
```

**Threaded Execution with Logging:**
```python
# Run agents in background thread while displaying logs
def run_with_logging(user_query, ...):
    log_queue = queue.Queue()
    # Worker thread executes agents
    # Main thread displays logs in real-time
```

### Extending the System

**Adding a New Agent:**

1. Create agent file in `agents/` directory
2. Inherit from `Agent` base class
3. Implement required methods
4. Register in appropriate orchestrator (PlanningAgent or EnsembleAgent)

Example:
```python
# agents/my_new_agent.py
from agents.agent import Agent

class MyNewAgent(Agent):
    name = "My New Agent"
    color = Agent.CYAN

    def __init__(self):
        self.log("My New Agent is initializing")
        # Setup code
        self.log("My New Agent is ready")

    def process(self, query: str):
        self.log(f"Processing: {query}")
        # Implementation
        return result
```

**Adding New Data Sources:**

1. Place source files in `data/documents/knowledge_base/`
2. Run: `python data_parsing_scripts/initialize_vectorstore_frontier.py`
3. Vectorstore automatically updates with new content

**Customizing Response Style:**

Edit system prompt in `agents/planning_agent.py`:
```python
system_prompt = """You are a helpful technical support assistant...
[Customize guidelines here]
"""
```



## Security & Privacy

### Current Security Measures

✓ **Authentication**: Google OAuth with domain restriction (@nysus.net, @nysus.com only)
✓ **SQL Injection Prevention**: Query pattern validation in MCP Agent (SELECT/EXEC/WITH only)
✓ **Least Privilege DB Access**: Recommended read-only database accounts
✓ **Session Management**: 24-hour timeout with secure session storage
✓ **Credential Protection**: Environment variables for secrets, no hardcoded credentials
✓ **Connection Isolation**: Each user session maintains separate database credentials

### Data Privacy Considerations

**What Data is Stored:**
- Chat history (in session, cleared on browser close)
- Cached questions in `memory.json` (local file)
- Uploaded PDFs (temporary, in-memory processing only)
- Database connection credentials (session only, not persisted)

**What is Sent to External Services:**
- OpenAI: User queries, ticket content, database schemas (for LLM processing)
- Google: Email address for authentication
- SQL Server: Generated SQL queries

**PII (Personally Identifiable Information):**
- ⚠️ **Current State**: No PII redaction implemented
- ⚠️ **Risk**: Ticket descriptions and database results may contain customer names, emails, phone numbers
- ✅ **Recommended**: Implement PII detection and masking before LLM processing

### Security Best Practices

**For Database Connections:**
```sql
-- Create dedicated read-only user
CREATE LOGIN naas_readonly WITH PASSWORD = 'SecurePassword123!';
CREATE USER naas_readonly FOR LOGIN naas_readonly;

-- Grant read-only permissions
EXEC sp_addrolemember 'db_datareader', 'naas_readonly';

-- Optionally grant execute on specific stored procedures
GRANT EXECUTE ON [dbo].[sp_GetProjectInfo] TO naas_readonly;
```

**For API Keys:**
- Never commit `.env` file to version control
- Rotate OpenAI API keys regularly
- Use separate API keys for dev/staging/production
- Implement rate limiting on OpenAI API calls

**For Production Deployment:**
- Enable HTTPS/TLS for all connections
- Use secrets management service (not `.env` file)
- Implement request logging and audit trails
- Add rate limiting per user
- Implement IP whitelisting if possible
- Regular security audits of dependencies: `pip-audit`

### Recommended Enhancements

1. **PII Redaction:**
   ```python
   # Before sending to OpenAI
   def redact_pii(text: str) -> str:
       # Mask emails, phone numbers, SSNs, credit cards
       text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
       text = re.sub(r'\b\d{3}-\d{3}-\d{4}\b', '[PHONE]', text)
       return text
   ```

2. **Audit Logging:**
   ```python
   # Log all database queries
   def log_query_audit(user_email, query, timestamp):
       # Write to audit log
       pass
   ```

3. **Data Retention Policy:**
   - Implement automatic cleanup of old chat histories
   - Limit `memory.json` size (e.g., last 1000 questions)
   - Purge uploaded PDFs after session ends

4. **Multi-Factor Authentication (MFA):**
   - Google OAuth supports MFA - encourage users to enable it
   - Consider additional security layer for production systems

### Compliance Considerations

**GDPR (if applicable):**
- Implement right to erasure (delete user data)
- Data processing agreement with OpenAI
- Document data flows and retention policies

**Industry Standards:**
- SOC 2 compliance for cloud deployments
- Regular penetration testing
- Vulnerability scanning of dependencies

### Incident Response

**If Credentials are Compromised:**
1. Immediately rotate affected credentials
2. Revoke OAuth tokens
3. Audit access logs for suspicious activity
4. Notify affected users if data breach occurred

**If API Keys are Exposed:**
1. Revoke and rotate OpenAI API keys immediately
2. Check OpenAI usage logs for unauthorized access
3. Update `.env` and secrets management systems
4. Review recent commits/logs for exposure source

## Roadmap & Future Enhancements

### Short-term (1-3 months)

- [ ] **PII Redaction System**: Automatically detect and mask sensitive information
- [ ] **Automated Testing**: Comprehensive unit and integration test suite
- [ ] **Performance Optimization**: Cache frequent queries, optimize embeddings
- [ ] **Enhanced Error Handling**: Better user-facing error messages
- [ ] **Query History Export**: Allow users to download their conversation history
- [ ] **Advanced Search Filters**: Filter tickets by date, company, status

### Medium-term (3-6 months)

- [ ] **Role-Based Access Control (RBAC)**: Different permissions for different user roles
- [ ] **Multi-tenant Support**: Separate vectorstores and databases per customer
- [ ] **Advanced Analytics Dashboard**: Usage metrics, popular queries, response quality
- [ ] **Feedback System**: Thumbs up/down on responses for continuous improvement
- [ ] **Custom Agent Training**: Fine-tune models on Nysus-specific data
- [ ] **Slack/Teams Integration**: Use NAAS from chat platforms

### Long-term (6-12 months)

- [ ] **Voice Interface**: Speech-to-text for hands-free queries
- [ ] **Proactive Notifications**: Alert users about similar open issues
- [ ] **Integration with Ticketing System**: Create/update tickets directly from NAAS
- [ ] **Knowledge Graph**: Structured representation of relationships between tickets
- [ ] **Automated Ticket Classification**: ML model to categorize and route new tickets
- [ ] **Multi-language Support**: Serve global teams in their preferred language

### Technical Debt & Improvements

- [ ] Migrate from `pyodbc` to async database driver for better performance
- [ ] Implement connection pooling for SQL Server
- [ ] Add retry logic for transient API failures
- [ ] Optimize Chroma vectorstore indexing
- [ ] Implement proper logging framework (replace print statements)
- [ ] Add comprehensive docstrings to all functions
- [ ] Refactor large functions in `nysus_chatbot.py` into smaller components
- [ ] Implement proper error handling hierarchy
- [ ] Add type hints throughout codebase
- [ ] Create CI/CD pipeline for automated testing and deployment

## Contributing

### How to Contribute

We welcome contributions from the Nysus engineering team and community!

**Getting Started:**

1. **Fork the Repository**
   ```powershell
   # Clone your fork
   git clone https://github.com/your-username/nysus_chatbot.git
   cd nysus_chatbot
   ```

2. **Create a Feature Branch**
   ```powershell
   git checkout -b feature/your-feature-name
   # or
   git checkout -b bugfix/issue-description
   ```

3. **Make Your Changes**
   - Follow existing code style and conventions
   - Add comments for complex logic
   - Update documentation if needed

4. **Test Your Changes**
   ```powershell
   # Run manual tests
   streamlit run nysus_chatbot.py

   # Run automated tests (when implemented)
   pytest tests/
   ```

5. **Commit and Push**
   ```powershell
   git add .
   git commit -m "Add feature: description of your changes"
   git push origin feature/your-feature-name
   ```

6. **Open a Pull Request**
   - Go to GitHub repository
   - Click "New Pull Request"
   - Provide clear description of changes
   - Reference any related issues
   - Wait for code review

### Coding Standards

**Python Style:**
- Follow PEP 8 guidelines
- Use meaningful variable names
- Maximum line length: 120 characters
- Use type hints where possible

**Docstring Format:**
```python
def function_name(param1: str, param2: int) -> dict:
    """
    Brief description of function.

    :param param1: Description of param1
    :param param2: Description of param2
    :return: Description of return value
    """
    pass
```

**Logging:**
```python
# Use agent logging pattern
self.log("Agent is performing action")  # For agent classes
logging.info("General information")     # For other modules
```

**Error Handling:**
```python
try:
    # Risky operation
    result = perform_operation()
except SpecificException as e:
    self.log(f"Error: {e}")
    # Provide user-friendly fallback
```

### Code Review Checklist

- [ ] Code follows PEP 8 style guidelines
- [ ] All functions have docstrings
- [ ] No hardcoded credentials or API keys
- [ ] Error handling is appropriate
- [ ] Logging is implemented for important actions
- [ ] No unnecessary dependencies added
- [ ] Comments explain "why" not "what"
- [ ] Variable names are descriptive
- [ ] Changes are backward compatible (or migration plan provided)
- [ ] Documentation updated if needed

### Reporting Issues

**Bug Reports:**
- Use GitHub Issues
- Include steps to reproduce
- Provide error messages and logs
- Specify environment (OS, Python version, dependencies)

**Feature Requests:**
- Describe the problem or use case
- Propose potential solution
- Explain benefits to users
- Consider implementation complexity

### Contact & Support

**For questions about NAAS or this repository:**
- Open a GitHub Discussion
- Contact the Nysus engineering team
- Email: [team email if applicable]

**For urgent security issues:**
- Do NOT open public issues
- Contact security team directly
- Provide detailed vulnerability description



# Google Authentication Setup Guide

This guide will walk you through setting up Google OAuth authentication for the Nysus Chatbot.

## Prerequisites

- Google Cloud Platform account
- Access to Google Cloud Console
- Python environment with pip

## Step 1: Install Required Packages

```powershell
pip install -r requirements_auth.txt
```

Or install individually:
```powershell
pip install google-auth-oauthlib google-auth google-api-python-client python-dotenv
```

## Step 2: Create Google Cloud Project and OAuth Credentials

### 2.1 Go to Google Cloud Console
1. Visit [Google Cloud Console](https://console.cloud.google.com/)
2. Sign in with your Google account

### 2.2 Create or Select a Project
1. Click on the project dropdown at the top of the page
2. Click "New Project"
3. Enter project name: `nysus-chatbot` (or any name you prefer)
4. Click "Create"

### 2.3 Enable Required APIs
1. Go to **APIs & Services** > **Library**
2. Search for "Google+ API" or "People API"
3. Click on it and click "Enable"

### 2.4 Configure OAuth Consent Screen
1. Go to **APIs & Services** > **OAuth consent screen**
2. Choose **Internal** (if you have Google Workspace) or **External**
3. Fill in the required information:
   - App name: `Nysus Chatbot`
   - User support email: your email
   - Developer contact information: your email
4. Click "Save and Continue"
5. On the Scopes page, you can skip (default scopes are sufficient)
6. Click "Save and Continue"
7. Review and click "Back to Dashboard"

### 2.5 Create OAuth 2.0 Credentials
1. Go to **APIs & Services** > **Credentials**
2. Click **Create Credentials** > **OAuth 2.0 Client IDs**
3. If prompted, configure the consent screen first (see step 2.4)
4. Application type: **Web application**
5. Name: `Nysus Chatbot Web Client`
6. **Authorized redirect URIs** - Add these URIs:
   - For local development: `http://localhost:8501`
   - For production: `https://your-production-domain.com`
7. Click **Create**
8. A dialog will appear with your **Client ID** and **Client Secret**
9. **IMPORTANT**: Copy these values immediately - you'll need them in the next step

## Step 3: Configure Environment Variables

### 3.1 Create .env File
1. Copy the `.env.example` file to `.env`:
   ```powershell
   Copy-Item .env.example .env
   ```

2. Open the `.env` file in a text editor

3. Replace the placeholder values with your actual credentials:
   ```env
   GOOGLE_CLIENT_ID=your_actual_client_id_here.apps.googleusercontent.com
   GOOGLE_CLIENT_SECRET=your_actual_client_secret_here
   REDIRECT_URI=http://localhost:8501
   ```

### 3.2 For Production Deployment
When deploying to production:
1. Update `REDIRECT_URI` to your production URL
2. Add the production URL to authorized redirect URIs in Google Cloud Console
3. Ensure `.env` file is **NOT** committed to version control (add to `.gitignore`)

## Step 4: Configure Allowed Email Domains

The authentication is configured to only allow users with `@nysus.net` or `@nysus.com` email addresses.

To modify allowed domains, edit `auth_config.py`:
```python
ALLOWED_DOMAINS = ["nysus.net", "nysus.com"]
```

## Step 5: Run the Application

```powershell
streamlit run nysus_chatbot.py
```

The application will:
1. Show a login page
2. Redirect to Google for authentication
3. Verify the email domain
4. Grant access only to authorized users

## Security Features

✓ **Domain Restriction**: Only `@nysus.net` and `@nysus.com` emails allowed
✓ **Secure OAuth 2.0**: Industry-standard authentication
✓ **Session Management**: 24-hour session timeout
✓ **No Password Storage**: Authentication handled by Google

## Troubleshooting

### Error: "Redirect URI mismatch"
- Ensure the redirect URI in Google Cloud Console exactly matches `REDIRECT_URI` in `.env`
- Don't forget to include the protocol (`http://` or `https://`)
- For Streamlit Cloud, use the full deployment URL

### Error: "Access Denied: Only @nysus.net..."
- User is trying to sign in with non-authorized email domain
- Verify the email address belongs to nysus.net or nysus.com domain

### Error: "Google OAuth credentials not configured"
- The `.env` file is missing or variables are empty
- Verify `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET` are set correctly

### Error: "This app isn't verified"
- This appears when using OAuth consent screen in "External" mode
- Click "Advanced" > "Go to [App Name] (unsafe)" to proceed (for testing)
- For production, complete Google's app verification process

## Additional Configuration

### Session Timeout
To modify session duration, edit `auth_config.py`:
```python
SESSION_MAX_AGE = 86400  # 24 hours in seconds
```

### Adding More Allowed Domains
Edit `auth_config.py`:
```python
ALLOWED_DOMAINS = ["nysus.net", "nysus.com", "example.com"]
```

## Production Deployment Notes

### Streamlit Cloud
1. Add environment variables in Streamlit Cloud settings (Secrets)
2. Update `REDIRECT_URI` to your Streamlit Cloud app URL
3. Add the Streamlit Cloud URL to authorized redirect URIs in Google Cloud Console

### Other Cloud Providers
1. Set environment variables through the provider's interface
2. Ensure SSL/TLS is enabled (HTTPS)
3. Update redirect URIs accordingly

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify all environment variables are set correctly
3. Ensure Google Cloud Console configuration matches your setup
4. Check Streamlit logs for detailed error messages

## License

This project does not include a license in the repository by default. Add a `LICENSE` file (e.g., Apache-2.0 or MIT) to indicate project licensing.

---

## Quick Reference Card

### Essential Commands

```powershell
# Setup
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install streamlit langchain langchain-openai langchain-chroma pyodbc python-dotenv PyPDF2 google-auth-oauthlib

# Run Application
streamlit run nysus_chatbot.py

# Rebuild Vectorstore
python data_parsing_scripts/initialize_vectorstore_frontier.py

# Check Dependencies
pip list
Get-OdbcDriver  # Verify ODBC Driver 17 for SQL Server
```

### Key Files Reference

| File | Purpose |
|------|---------|
| `nysus_chatbot.py` | Main Streamlit UI application |
| `chatbot_agent_framework.py` | Agent orchestration and memory management |
| `auth.py` | Google OAuth authentication |
| `auth_config.py` | Authentication settings |
| `memory.json` | Cached questions (created at runtime) |
| `.env` | Environment variables (not in repo) |
| `agents/planning_agent.py` | Response synthesis (OpenAI GPT-4o-mini) |
| `agents/scanner_agent.py` | Cache retrieval (Ollama Llama3.2) |
| `agents/frontier_agent.py` | RAG search (OpenAI embeddings) |
| `agents/mcp_agent.py` | SQL generation and execution |
| `agents/ensemble_agent.py` | Coordinates Frontier + MCP |
| `data/documents/knowledge_base/` | Source markdown for vectorstore |
| `data/documents/os_tickets_vectorstore/frontier/` | Chroma vectorstore database |

### Default Configuration

| Setting | Value |
|---------|-------|
| **OpenAI Model** | gpt-4o-mini |
| **Ollama Model** | llama3.2 |
| **Embeddings** | OpenAI text-embedding-3-large (1536d) |
| **Vectorstore K** | 20 retrieval, 10 final |
| **Session Timeout** | 24 hours |
| **Allowed Domains** | @nysus.net, @nysus.com |
| **ODBC Driver** | ODBC Driver 17 for SQL Server |
| **Default Port** | 8501 (Streamlit) |

### Environment Variables

```env
# Required
OPENAI_API_KEY=sk-...
GOOGLE_CLIENT_ID=...apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=...
REDIRECT_URI=http://localhost:8501

# Optional
OLLAMA_API_URL=http://localhost:11434
```

### Agent Color Codes

| Agent | Color | Terminal Code |
|-------|-------|---------------|
| Planning Agent | 🟢 Green | `\033[32m` |
| Scanner Agent | 🔵 Cyan | `\033[36m` |
| Frontier Agent | 🔵 Blue | `\033[34m` |
| MCP Agent | 🟣 Magenta | `\033[35m` |
| Ensemble Agent | 🟡 Yellow | `\033[33m` |
| Specialist Agent | 🔴 Red | `\033[31m` |

### Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| "This app isn't verified" | Click Advanced → Go to app (development only) |
| "Redirect URI mismatch" | Match `.env` REDIRECT_URI with Google Cloud Console |
| SQL connection fails | Check ODBC driver: `Get-OdbcDriver` |
| Vectorstore empty | Run `initialize_vectorstore_frontier.py` |
| OpenAI rate limit | Reduce request frequency or upgrade API plan |
| Session expired | Re-authenticate (automatic redirect) |

### Performance Benchmarks

| Operation | Expected Time |
|-----------|--------------|
| Authentication | 1-3 seconds |
| Ticket search (RAG) | 2-5 seconds |
| SQL query generation | 3-7 seconds |
| PDF upload processing | 5-15 seconds |
| Full query response | 8-15 seconds |
| Vectorstore rebuild | 5-10 minutes |

### API Cost Estimates (OpenAI)

| Operation | Approximate Cost |
|-----------|------------------|
| Single query (with synthesis) | $0.01-0.03 |
| PDF processing (10 pages) | $0.005-0.01 |
| Vectorstore rebuild (full) | $2-5 |
| Daily usage (50 queries) | $0.50-1.50 |
| Monthly (1000 queries) | $10-30 |

*Costs based on OpenAI GPT-4o-mini and text-embedding-3-large pricing as of 2024*

---

## Project Status

**Current Version**: 1.0 (Beta)
**Last Updated**: November 2025
**Status**: ✅ Active Development
**Production Ready**: ⚠️ Pilot/Testing Phase

### Known Limitations

- No PII redaction implemented
- Limited to single-instance deployment
- No automated tests
- Memory stored in local file only
- No audit logging
- Session state not shared across instances

### Stability Notes

**Stable Components:**
- ✅ Google OAuth authentication
- ✅ RAG ticket retrieval
- ✅ Database connectivity
- ✅ Multi-agent orchestration
- ✅ PDF upload and processing

**Experimental/Beta:**
- ⚠️ MCP Agent SQL generation (validate queries manually)
- ⚠️ Scanner Agent cache selection (accuracy varies)
- ⚠️ PDF context integration (depends on document format)

---

## Acknowledgments

Built by the Nysus engineering team using:
- **Streamlit** - Web application framework
- **LangChain** - LLM orchestration
- **OpenAI** - GPT-4o-mini and embeddings
- **Ollama** - Local LLM inference
- **Chroma** - Vector database
- **Google OAuth** - Authentication

Special thanks to all contributors and testers who helped shape NAAS into a valuable tool for the MES support team.

---

## Contact

**For questions about NAAS or this repository:**
- Open a GitHub Issue or Discussion
- Contact the Nysus engineering team
- Repository: https://github.com/lehoangbaoduy/nysus_chatbot

**For urgent security issues:**
- Do NOT open public issues
- Contact security team directly

---

*README last updated: November 27, 2025*
*NAAS - Making MES support smarter, faster, and more accessible* 🚀
