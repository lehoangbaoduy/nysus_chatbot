from typing import Dict, List, Optional
from agents.agent import Agent
from agents.frontier_agent import FrontierAgent
# from agents.open_source_agent import OpenSourceAgent
from agents.mcp_agent import MCPAgent

class EnsembleAgent(Agent):

    name = "Ensemble Agent"
    color = Agent.YELLOW

    def __init__(self, collection=None, db_connection_params: Optional[Dict] = None):
        """
        Create an instance of Ensemble, by creating each of the models
        And loading the weights of the Ensemble
        """
        self.log("Initializing Ensemble Agent")
        self.log(f"Received db_connection_params: {db_connection_params is not None}")
        self.frontier = FrontierAgent(collection)
        self.mcp = MCPAgent(db_connection_params)
        self.log("Ensemble Agent is ready")

    def answer_question(self, user_query: str, chat_history: List = [], n_results: int = 5, schema: str = None, uploaded_files: List = [], agent_selection: List[str] = None) -> Dict:
        """
        Answer a user's question by finding relevant tickets from the knowledge base.
        Or trying to use MCP server when connect to our SQL server to answer unanswered questions.
        Orchestrates the workflow by delegating to FrontierAgent for RAG search.
        :param user_query: the user's question
        :param chat_history: previous conversation context
        :param n_results: number of results to return
        :param schema: optional pre-fetched database schema
        :param uploaded_files: list of uploaded files
        :param agent_selection: list of selected agents to use (empty = all agents)
        :return: a dict containing relevant_tickets and optionally mcp_response
        """
        # Default to all agents if none selected
        if not agent_selection:
            agent_selection = ["Tickets", "Uploaded Documents", "SQL Server"]

        response = {
            'relevant_tickets': [],
            'relevant_document_content': None,
            'relevant_document_sources': [],
            'mcp_response': None
        }

        self.log(f"Ensemble Agent is processing user query: '{user_query[:50]}...'")
        self.log(f"Ensemble Agent received agent_selection: {agent_selection}")
        # Check if Tickets agent should be used
        if "Tickets" in agent_selection:
            self.log("Ensemble Agent delegating ticket search to Frontier Agent")
            # Use FrontierAgent to find relevant tickets from ChromaDB
            relevant_tickets = self.frontier.find_relevant_tickets(user_query, n_results)

            if relevant_tickets:
                response['relevant_tickets'] = relevant_tickets
        else:
            self.log("Ticket search disabled by agent selection")

        # Check if Uploaded Documents agent should be used
        if "Uploaded Documents" in agent_selection and uploaded_files:
            self.log("Ensemble Agent delegating document search to Frontier Agent")
            # Use FrontierAgent to find relevant uploaded files content
            document_response = self.frontier.find_relevant_uploaded_content(user_query, uploaded_files)

            if document_response:
                if isinstance(document_response, dict):
                    response['relevant_document_content'] = document_response.get('content')
                    response['relevant_document_sources'] = document_response.get('sources', [])
                else:
                    # Fallback for old format (just string)
                    response['relevant_document_content'] = document_response
        else:
            if "Uploaded Documents" not in agent_selection:
                self.log("Document search disabled by agent selection")
            elif not uploaded_files:
                self.log("No uploaded files provided for document search")

        # Check if SQL Server agent should be used
        if "SQL Server" in agent_selection:
            self.log("Ensemble Agent delegating to MCP Agent for database query")
            mcp_result = self.mcp.answer_question_with_database(user_query, chat_history, schema)

            if mcp_result.get('success'):
                self.log(f"MCP Agent successfully queried database: {mcp_result.get('database')}")
            else:
                self.log(f"MCP Agent query failed or no results: {mcp_result.get('error', 'Unknown error')}")

            response['mcp_response'] = mcp_result
        else:
            self.log("SQL Server query disabled by agent selection")

        return response