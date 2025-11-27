import logging
import queue
import threading
import time
import base64
import pyodbc
import concurrent.futures
import streamlit as st
from typing import List, Dict
from log_utils import reformat
from chatbot_agent_framework import ChatbotAgentFramework
from langchain_core.messages import AIMessage, HumanMessage
from auth import check_authentication, show_login_page, show_user_info

class QueueHandler(logging.Handler):
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.put(self.format(record))

# ------------------- Utility Functions -------------------
def setup_logging(log_queue):
    handler = QueueHandler(log_queue)
    formatter = logging.Formatter(
        "[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S %z",
    )
    handler.setFormatter(formatter)
    logger = logging.getLogger("logging")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

def load_css(file_path):
    """Load CSS from external file"""
    with open(file_path, 'r') as f:
        return f'<style>{f.read()}</style>'

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def group_databases_and_tables(matching_databases: List[Dict[str, str]]) -> Dict[str, List[str]]:
    """
    Group tables by their parent database.

    Args:
        matching_databases: List of dicts with 'database' and 'table' keys

    Returns:
        Dict mapping database names to lists of table names
    """
    grouped = {}
    for item in matching_databases:
        db_name = item.get("database", "Unknown")
        table_name = item.get("table", "Unknown")

        if db_name not in grouped:
            grouped[db_name] = []

        # Only add unique tables
        if table_name not in grouped[db_name]:
            grouped[db_name].append(table_name)

    # Sort tables alphabetically within each database
    for db_name in grouped:
        grouped[db_name].sort()

    return grouped

def set_background(png_file, opacity):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = f'''
    <style>
    .stApp {{
    background: none !important;
    }}

    [data-testid="stAppViewContainer"]::before {{
        content: "";
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        opacity: {opacity};
        z-index: -1;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

class App:

    def __init__(self):
        # Initialize agent framework in session state if not exists
        if 'agent_framework' not in st.session_state:
            st.session_state.agent_framework = None

    def get_agent_framework(self, db_connection_params=None):
        if st.session_state.agent_framework is None:
            # Create framework once with connection params from session state if available
            params = db_connection_params or st.session_state.get("db_connection_params", None)
            st.session_state.agent_framework = ChatbotAgentFramework(params)
            st.session_state.agent_framework.init_agents_as_needed()
        else:
            # Update connection params in existing framework if needed
            params = st.session_state.get("db_connection_params", None)
            st.session_state.agent_framework.update_db_connection_params(params)
        return st.session_state.agent_framework

    def run(self):
        # ------------------- Authentication Check -------------------
        # Check if user is authenticated before showing the app
        if not check_authentication():
            show_login_page()
            return

        def do_run(user_query: str, chat_history: list, uploaded_files: list, agent_framework, schema=None):
            """Run the agent framework and return new opportunities"""
            # Use the passed agent framework
            result = agent_framework.run(user_query, chat_history, uploaded_files, schema)

            # Handle response from agent framework
            # Result can be either a Dict (from Planning/Ensemble Agent) or List (from memory)
            if isinstance(result, dict):
                # Response from Planning Agent with tickets, MCP data, and/or recently asked questions
                relevant_tickets = result.get('relevant_tickets', [])
                mcp_response = result.get('mcp_response', None)
                natural_response = result.get('natural_response', None)
                recently_asked_questions = result.get('recently_asked_questions', [])

                response_text = ""
                matching_tickets = []
                matching_databases = []

                # Extract ticket numbers for sidebar
                if relevant_tickets:
                    for ticket in relevant_tickets:
                        matching_tickets.append(ticket)

                # Extract database info for sidebar
                if mcp_response and mcp_response.get('database'):
                    matching_databases.append({
                        'database': mcp_response['database'] if 'database' in mcp_response else 'N/A',
                        'table': mcp_response['table'] if 'table' in mcp_response else 'N/A'
                    })

                # Use natural LLM-generated response if available
                # This response already synthesizes tickets, questions, and database results
                if natural_response:
                    response_text = natural_response
                # Fallback to structured response if LLM generation failed
                elif recently_asked_questions:
                    # Show recently asked questions if they exist
                    response_text = f"I found {len(recently_asked_questions)} similar question(s) asked recently:\n\n"
                    for i, question in enumerate(recently_asked_questions, 1):
                        if hasattr(question, 'question'):
                            response_text += f"{i}. {question.question}\n"
                            if hasattr(question, 'answer') and question.answer:
                                response_text += f"   Answer: {question.answer[:200]}...\n\n"
                elif relevant_tickets or (mcp_response and mcp_response.get('success')):
                    # Show relevant tickets and/or database response
                    if relevant_tickets:
                        response_text = f"I found {len(relevant_tickets)} relevant tickets that might help answer your question.\n\n"
                        for ticket in relevant_tickets:
                            if hasattr(ticket, 'ticket_number'):
                                response_text += f"**Ticket #{ticket.ticket_number}**: {ticket.subject if hasattr(ticket, 'subject') else 'No subject'}\n"
                                if hasattr(ticket, 'description'):
                                    response_text += f"Description: {ticket.description}\n\n"
                                if hasattr(ticket, 'score'):
                                    response_text += f"Relevance Score: {ticket.score}\n\n"

                    if mcp_response and mcp_response.get('success'):
                        if relevant_tickets:
                            response_text += "\n\n---\n\n**Additional information from database:**\n\n"
                        response_text += mcp_response.get('user_message', '')
                else:
                    response_text = "I couldn't find any relevant information for your question. Could you please rephrase or provide more details?"

                if not response_text:
                    response_text = "I couldn't find any relevant information for your question. Could you please rephrase or provide more details?"

                return {
                    "response": response_text,
                    "matching_tickets": matching_tickets,
                    "matching_databases": matching_databases
                }

        def run_with_logging(user_query: str, chat_history: list, uploaded_files: list, agent_framework, schema=None):
            """Run agent with logging capability"""
            log_queue = queue.Queue()
            result_queue = queue.Queue()
            setup_logging(log_queue)

            def worker():
                result = do_run(user_query, chat_history, uploaded_files, agent_framework, schema)
                result_queue.put(result)

            thread = threading.Thread(target=worker)
            thread.start()

            # Update logs and results
            while True:
                try:
                    message = log_queue.get_nowait()

                    if 'log_data' not in st.session_state:
                        st.session_state.log_data = []

                    st.session_state.log_data.append(reformat(message))
                except queue.Empty:
                    try:
                        final_result = result_queue.get_nowait()
                        return final_result
                    except queue.Empty:
                        time.sleep(0.1)

        # ------------------- Load Custom CSS -------------------------
        dual_sidebar_css = load_css('styles/styles.css')

        # ------------------- Streamlit UI Settings -------------------
        st.set_page_config(page_title="Nysus Chatbot (v0.0.1)", page_icon="data/image/logo.ico", layout="wide")
        st.markdown(dual_sidebar_css, unsafe_allow_html=True)
        st.title("NAAS - Nysus Automated Assistant for MES")

        # Initialize session state
        if 'agent_framework' not in st.session_state:
            st.session_state.agent_framework = None
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [AIMessage(content="Hello! I'm a Nysus automated assistant. Ask me anything about MES!")]
        if "matching_tickets" not in st.session_state:
            st.session_state.matching_tickets = []
        if "matching_databases" not in st.session_state:
            st.session_state.matching_databases = []
        if 'db_schema' not in st.session_state:
            st.session_state.db_schema = None
        if 'show_connection_message' not in st.session_state:
            st.session_state.show_connection_message = None

        # ------------------- Left Sidebar - Settings -------------------
        with st.sidebar:
            # Display user info at top of sidebar
            show_user_info()

            st.markdown("---")

            # Show persistent connection status
            is_connected = 'db_connection_params' in st.session_state and st.session_state.db_connection_params is not None
            if is_connected:
                st.subheader("SQL Server Connection Status: 🟢")
            else:
                st.subheader("SQL Server Connection Status: 🔴")

            st.text_input("Host", value=r"np:\\.\pipe\LOCALDB#94C6051A\tsql\query", key="Host")
            st.text_input("Port", value="1433", key="Port")
            st.text_input("User", value="nysususer", key="User")
            st.text_input("Password", type="password", value="nysus2444", key="Password")

            # Create two columns for the buttons
            col1, col2 = st.columns(2)

            with col1:
                connect_button = st.button("Connect to SQL Server")

            with col2:
                disconnect_button = st.button("Disconnect from SQL Server", disabled=not is_connected)

            # Show connection/disconnection messages if they exist
            if st.session_state.show_connection_message:
                if st.session_state.show_connection_message['type'] == 'success':
                    st.success(st.session_state.show_connection_message['message'])
                elif st.session_state.show_connection_message['type'] == 'error':
                    st.error(st.session_state.show_connection_message['message'])
                st.session_state.show_connection_message = None

            if connect_button:
                with st.spinner("Testing SQL Server connection..."):
                    try:
                        # Test basic connection
                        driver = "ODBC Driver 17 for SQL Server"
                        odbc_str = (
                            f"DRIVER={{{driver}}};"
                            f"SERVER={st.session_state['Host']};"
                            f"UID={st.session_state['User']};"
                            f"PWD={st.session_state['Password']};"
                            "Trusted_Connection=no;"
                        )
                        conn = pyodbc.connect(odbc_str)
                        cursor = conn.cursor()
                        cursor.execute("""
                            SELECT name FROM sys.databases
                            WHERE name NOT IN ('master', 'tempdb', 'model', 'msdb');
                        """)
                        databases = [row[0] for row in cursor.fetchall()]
                        conn.close()

                        if not databases:
                            st.warning("Connection successful, but no user databases found.")
                        else:
                            # Store connection parameters for agent framework
                            st.session_state.db_connection_params = {
                                'host': st.session_state["Host"],
                                'port': st.session_state["Port"],
                                'user': st.session_state["User"],
                                'password': st.session_state["Password"],
                                'driver': driver
                            }

                            # Get or create agent framework
                            if st.session_state.agent_framework is None:
                                st.session_state.agent_framework = ChatbotAgentFramework(st.session_state.db_connection_params)
                                st.session_state.agent_framework.init_agents_as_needed()
                            else:
                                st.session_state.agent_framework.update_db_connection_params(st.session_state.db_connection_params)

                            # Fetch and cache the schema
                            with st.spinner("Fetching database schema..."):
                                schema = st.session_state.agent_framework.planner.ensemble.mcp.get_schema()
                                if "❌" not in schema and "No user databases" not in schema:
                                    st.session_state.db_schema = schema
                                    st.session_state.show_connection_message = {
                                        'type': 'success',
                                        'message': f"✅ Connected to SQL Server with {len(databases)} database(s) - Schema cached"
                                    }
                                    st.rerun()
                                else:
                                    st.session_state.show_connection_message = {
                                        'type': 'error',
                                        'message': "❌ Connected but failed to fetch schema"
                                    }

                    except Exception as e:
                        st.session_state.show_connection_message = {
                            'type': 'error',
                            'message': f"❌ Connection failed: {e}"
                        }

            if disconnect_button:
                # Clear all database-related session state
                if 'db_connection_params' in st.session_state:
                    del st.session_state.db_connection_params
                if 'db_schema' in st.session_state:
                    del st.session_state.db_schema
                if 'agent_framework' in st.session_state:
                    st.session_state.agent_framework = None
                if 'matching_databases' in st.session_state:
                    st.session_state.matching_databases = []

                st.session_state.show_connection_message = {
                    'type': 'error',
                    'message': "❌ Disconnected from SQL Server"
                }
                st.rerun()

            st.subheader("Upload additional document files")
            user_uploaded_files = st.file_uploader("", type=["csv", "xlsx", "txt", "pdf"], accept_multiple_files=True)

            st.subheader("Background Image Selector")
            backgrounds = {"Nysus Tiles": "data/image/background.png", "Nysus More Tiles": "data/image/background2.png"}
            choice = st.selectbox("Choose a background:", list(backgrounds.keys()))
            opacity = st.sidebar.slider("Background opacity", 0.0, 1.0, 0.1, 0.05)
            st.image(backgrounds[choice], width=150, caption=f"Preview: {choice}")
            if st.button("Set background"):
                st.session_state.background_path = backgrounds[choice]
            if "background_path" in st.session_state:
                set_background(st.session_state.background_path, opacity)

        # ------------------- Main Layout - Chat and Right Sidebar -------------------
        col_chat, col_right = st.columns([3, 1])

        # Chat Section (Middle Column)
        with col_chat:
            for message in st.session_state.chat_history:
                with st.chat_message("AI" if isinstance(message, AIMessage) else "Human",
                                    avatar="data/image/logo.ico" if isinstance(message, AIMessage) else "🦖"):
                    st.markdown(message.content)

            user_query = st.chat_input("Type a message...")
            if user_query and user_query.strip():
                # Add human message to chat history
                st.session_state.chat_history.append(HumanMessage(content=user_query))
                with st.chat_message("Human", avatar="🦖"):
                    st.markdown(user_query)

                # Placeholder for AI message
                with st.chat_message("AI", avatar="data/image/logo.ico"):
                    message_placeholder = st.empty()

                    # Capture the values we need before threading
                    chat_history = st.session_state.chat_history.copy()  # Make a copy!
                    # Get or create agent framework BEFORE threading
                    current_agent_framework = self.get_agent_framework()
                    # Get cached schema if available
                    cached_schema = st.session_state.get('db_schema', None)

                    # THIS IS THE PLACE FOR ALL BACKEND AI STUFF STARTS
                    def get_response_wrapper():
                        return run_with_logging(
                            user_query,
                            chat_history,
                            user_uploaded_files,
                            current_agent_framework,
                            cached_schema
                        )

                    # Start getting response in background
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(get_response_wrapper)

                        # Animate while waiting
                        counter = 0
                        while not future.done():
                            dots = "." * ((counter % 3) + 1)
                            message_placeholder.markdown(f"💬 **Assistant is thinking{dots}**")
                            time.sleep(0.3)
                            counter += 1

                        result = future.result()

                    # Handle response as dict
                    if isinstance(result, dict):
                        response = result["response"]
                        st.session_state.matching_tickets = result.get("matching_tickets", [])
                        st.session_state.matching_databases = result.get("matching_databases", [])
                    else:
                        response = result
                        st.session_state.matching_tickets = []
                        st.session_state.matching_databases = []

                    # Display response with streaming effect
                    displayed_text = ""
                    for char in response:
                        displayed_text += char
                        message_placeholder.markdown(displayed_text)
                        time.sleep(0.01)

                    st.session_state.chat_history.append(AIMessage(content=response))

        # Right Sidebar - Matching Info
        with col_right:
            st.markdown('<div class="sidebar-section-header">🎫 Matching Tickets</div>', unsafe_allow_html=True)
            if st.session_state.matching_tickets:
                # Check if any ticket has a URL
                has_any_url = any(hasattr(ticket, 'url') and ticket.url for ticket in st.session_state.matching_tickets)

                if not has_any_url:
                    st.info("No matching tickets found yet. Ask a question to see related tickets!")
                else:
                    # Sort tickets by score in descending order (highest similarity first)
                    sorted_tickets = sorted(
                        st.session_state.matching_tickets,
                        key=lambda t: t.score if hasattr(t, 'score') and t.score is not None else 0,
                        reverse=True
                    )

                    for ticket in sorted_tickets:
                        ticket_number = str(ticket.ticket_number) if hasattr(ticket, 'ticket_number') else "Unknown"
                        ticket_url = ticket.url if hasattr(ticket, 'url') and ticket.url else None

                        if ticket_url:
                            st.markdown(
                                f'<div class="ticket-item"><a href="{ticket_url}" target="_blank">🎫 Ticket #{ticket_number}</a></div>',
                                unsafe_allow_html=True
                            )

                            # Display similarity score with progress bar if available
                            if ticket.score is not None:
                                st.progress(ticket.score, text=f"Similarity: {ticket.score * 100:.1f}%")

                            # Add a small spacing between tickets
                            st.markdown('<div style="margin-bottom: 0.1rem;"></div>', unsafe_allow_html=True)
            else:
                st.info("No matching tickets found yet. Ask a question to see related tickets!")

            st.markdown('<div class="sidebar-section-header">🗄️ Related Databases & Tables</div>', unsafe_allow_html=True)
            if st.session_state.matching_databases:
                # Group tables by database
                grouped_data = group_databases_and_tables(st.session_state.matching_databases)

                for db_name, tables in grouped_data.items():
                    # Build the table list HTML
                    tables_html = ""
                    for table in tables:
                        tables_html += f'<div style="margin-left: 1rem; padding: 0.25rem 0; color: #e8f4f8;">└─ {table}</div>'

                    # Display database with all its tables
                    st.markdown(
                        f'''<div class="db-item">
                            <div style="font-weight: 600; margin-bottom: 0.25rem;">📊 {db_name}</div>
                            {tables_html}
                        </div>''',
                        unsafe_allow_html=True
                    )
            else:
                st.info("No database queries executed yet. Connected databases will appear here after queries.")

if __name__=="__main__":
    App().run()
