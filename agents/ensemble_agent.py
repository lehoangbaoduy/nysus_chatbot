from typing import Dict, List, Optional
from agents.agent import Agent
from agents.tickets import Ticket
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

    def answer_question(self, user_query: str, chat_history: List = [], n_results: int = 5, schema: str = None) -> Dict:
        """
        Answer a user's question by finding relevant tickets from the knowledge base.
        Or trying to use MCP server when connect to our SQL server to answer unanswered questions.
        Orchestrates the workflow by delegating to FrontierAgent for RAG search.
        :param user_query: the user's question
        :param chat_history: previous conversation context
        :param n_results: number of results to return
        :param schema: optional pre-fetched database schema
        :return: a dict containing relevant_tickets and optionally mcp_response
        """
        response = {
            'relevant_tickets': [],
            'mcp_response': None
        }

        self.log(f"Ensemble Agent is processing user query: '{user_query[:50]}...'")
        self.log("Ensemble Agent delegating ticket search to Frontier Agent")

        # Use FrontierAgent to find relevant tickets from ChromaDB
        relevant_tickets = self.frontier.find_relevant_tickets(user_query, n_results)
        # relevant_tickets_response = self.frontier.answer_question_with_rag(user_query, chat_history)

        if relevant_tickets:
            response['relevant_tickets'] = relevant_tickets

        # Always call MCP agent for database query (even if tickets found)
        self.log("Ensemble Agent delegating to MCP Agent for database query")
        mcp_result = self.mcp.answer_question_with_database(user_query, chat_history, schema)

        if mcp_result.get('success'):
            self.log(f"MCP Agent successfully queried database: {mcp_result.get('database')}")
        else:
            self.log(f"MCP Agent query failed or no results: {mcp_result.get('error', 'Unknown error')}")

        response['mcp_response'] = mcp_result

        return response