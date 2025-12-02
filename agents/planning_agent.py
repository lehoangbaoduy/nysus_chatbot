from typing import Optional, List, Dict
from agents.agent import Agent
from agents.base_classes import Company, Ticket
from agents.scanner_agent import ScannerAgent
from agents.ensemble_agent import EnsembleAgent
from openai import OpenAI

class PlanningAgent(Agent):

    name = "Planning Agent"
    color = Agent.GREEN
    MODEL = "gpt-4o-mini"

    def __init__(self, collection=None, db_connection_params: Optional[Dict] = None):
        """
        Create instances of the 3 Agents that this planner coordinates across
        """
        self.log("Planning Agent is initializing")
        self.log(f"Received db_connection_params: {db_connection_params is not None}")
        self.scanner = ScannerAgent()
        self.ensemble = EnsembleAgent(collection, db_connection_params)
        self.client = OpenAI()
        self.log("Planning Agent is ready")

    def synthesize_response(self, user_query: str, tickets: List[Ticket], mcp_response: Optional[Dict], recently_asked_questions: List[str], relevant_document_content: str = None, chat_history: List = []) -> str:
        """
        Use LLM to generate a natural, conversational response based on retrieved information
        :param user_query: The user's original question
        :param tickets: List of relevant tickets
        :param mcp_response: Response from MCP database query
        :param recently_asked_questions: List of recently asked similar questions
        :param pdf_context: Context extracted from uploaded PDF files
        :param chat_history: Previous conversation context
        :return: Natural language response string
        """
        self.log("Synthesizing natural response using LLM")

        # Build context from uploaded PDF files
        relevant_document_context = ""
        if relevant_document_content:
            relevant_document_context = "\n\n=== Context from Uploaded Documents ===\n"
            relevant_document_context += relevant_document_content[:2000] + "..." if len(relevant_document_content) > 2000 else relevant_document_content
            relevant_document_context += "\n=== End of Document Context ===\n"
            self.log(f"Including PDF context in synthesis ({len(relevant_document_content)} chars)")

        # Build context from recently asked questions
        questions_context = ""
        if recently_asked_questions:
            questions_context = "\n\n=== Recently Asked Similar Questions ===\n"
            for i, question in enumerate(recently_asked_questions, 1):
                if hasattr(question, 'question'):
                    questions_context += f"\n{i}. {question.question}\n"
                    if hasattr(question, 'answer'):
                        answer_preview = question.answer[:200] + "..." if len(question.answer) > 200 else question.answer
                        questions_context += f"   Answer: {answer_preview}\n"
                else:
                    questions_context += f"\n{i}. {question}\n"

        # Build context from tickets
        ticket_context = ""
        if tickets:
            ticket_context = "\n\nRelevant support tickets found:\n"
            for i, ticket in enumerate(tickets, 1):  # Limit to top 5 for context
                ticket_context += f"\n{i}. Ticket #{ticket.ticket_number} - {ticket.subject}\n"
                ticket_context += f"   Company: {ticket.company}\n"
                if ticket.author:
                    ticket_context += f"   Author: {ticket.author}\n"
                if ticket.created_at:
                    ticket_context += f"   Created At: {ticket.created_at}\n"
                if ticket.updated_at:
                    ticket_context += f"   Updated At: {ticket.updated_at}\n"
                if ticket.score:
                    ticket_context += f"   Relevance Score: {ticket.score}\n"
                # Get first 300 chars of description
                desc_preview = ticket.description[:300] + "..." if len(ticket.description) > 300 else ticket.description
                ticket_context += f"   Description: {desc_preview}\n"

        # Build context from database query
        db_context = ""
        if mcp_response and mcp_response.get('success'):
            db_context = f"\n\nDatabase query results:\n"
            db_context += f"Database: {mcp_response.get('database')}\n"
            db_context += f"Query: {mcp_response.get('sql_query')}\n"
            db_context += f"Explanation: {mcp_response.get('explanation')}\n"

        # Build chat history context
        history_context = ""
        if chat_history and len(chat_history) > 1:
            history_context = "\n\nRecent conversation:\n"
            for msg in chat_history[-6:]:  # Last 3 exchanges
                role = "User" if hasattr(msg, 'content') and msg.__class__.__name__ == "HumanMessage" else "Assistant"
                content = msg.content[:150] + "..." if len(msg.content) > 150 else msg.content
                history_context += f"{role}: {content}\n"

        system_prompt = """You are a helpful technical support assistant for a manufacturing execution system (MES) company.
        Your role is to provide clear, conversational answers based on RAG responses from ticket knowledge base, database information, cached questions, and uploaded documents.

        Guidelines:
        - Be conversational and natural, not robotic
        - If context from uploaded PDF documents is provided, also reference it in your answer.
        - If recently asked similar questions are provided, acknowledge them and reference relevant ones
        - If a RAG system response is provided, it contains the score of relevance for each ticket, try your best to convert to verbal description when mentioning relevance.
        - Summarize technical information in an accessible way.
        - If multiple tickets are relevant, briefly mention how they relate or what common themes exist
        - If MCP response results are provided, ALWAYS show the database name, the executed SQL query (put it in a separate code block), and a brief explanation of the results.
        - If MCP response query IS NOT PROVIDED, do NOT mention anything about database query, just tell the user that MCP needs database connection to produce SQL query.
        - Stay focused on the user's question
        - If information is incomplete, acknowledge it naturally
        - Use a friendly, professional tone
        """

        user_prompt = f"""User Question: {user_query}
        {history_context}
        {relevant_document_context}
        {questions_context}
        {ticket_context}
        {db_context}

        Please provide a natural, helpful response to the user's question based on the information above.

        IMPORTANT:
        - If uploaded document context is provided, also reference it in your answer.
        - If a RAG response is provided, use it as the main source for ticket information
        - Synthesize the information into a coherent, conversational answer
        - Don't just repeat the RAG response - enhance it with database results and document context if available
        - If similar questions were asked recently, mention that and use their context to inform your response
        """

        try:
            response = self.client.chat.completions.create(
                model=self.MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=800
            )

            natural_response = response.choices[0].message.content
            self.log("Successfully generated natural response")
            return natural_response

        except Exception as e:
            self.log(f"Error generating natural response: {e}")
            # Fallback to simple concatenation if LLM fails
            fallback = "I found some relevant information for your question.\n\n"
            if relevant_document_context:
                fallback += "Found relevant information in your uploaded documents.\n"
            if recently_asked_questions:
                fallback += f"Found {len(recently_asked_questions)} similar question(s) asked recently.\n"
            if tickets:
                fallback += f"Found {len(tickets)} relevant ticket(s).\n"
            if mcp_response and mcp_response.get('success'):
                fallback += f"Also queried database: {mcp_response.get('database')}\n"
            return fallback

    def plan(self, memory: List[str] = [], user_query: str = "", chat_history: List = [], uploaded_files: List = [], schema: str = None) -> Optional[List]:
        """
        Run the full workflow:
        1. Use the ScannerAgent to see if the most recently asked questions match the user question (trying cached approach first)
        2. Use the EnsembleAgent to answer the user question based on tickets RAG approach
        3. Use the MessagingAgent to send a notification of Companys ----- not need for now
        :param memory: a list of cached questions that have been surfaced in the past
        :param user_query: the user's question
        :param chat_history: previous conversation context
        :param uploaded_files: any files uploaded by the user
        :param schema: optional pre-fetched database schema
        :return: a dict with response information including natural_response, rag_response, mcp_response
        """
        response = {
            'recently_asked_questions': [],
            'rag_response': None,
            'rag_chunks': [],
            'mcp_response': None,
            'natural_response': ""
        }

        self.log("Planning Agent is delegating to Scanner Agent to find recently asked questions")
        scanner_response = self.scanner.scan(memory=memory, user_query=user_query)

        # Extract PDF context from scanner response if available
        if scanner_response and scanner_response.questions:
            self.log(f"Planning Agent received {len(scanner_response.questions)} questions from Scanner")

            self.log("Planning Agent has completed a run")
            response['recently_asked_questions'] = scanner_response.questions
        else:
            self.log("No relevant recently asked questions found. Asking Ensemble Agent to search knowledge base.")

            # Use EnsembleAgent to answer the question by searching ChromaDB and potentially MCP server
            self.log("Planning Agent is delegating to Ensemble Agent to answer the user question")
            ensemble_response = self.ensemble.answer_question(user_query, chat_history, n_results=5, schema=schema, uploaded_files=uploaded_files)

            relevant_tickets = ensemble_response.get('relevant_tickets', None)
            if relevant_tickets:
                self.log(f"Planning Agent received {len(relevant_tickets)} relevant tickets from Ensemble Agent")
                response['relevant_tickets'] = relevant_tickets

            relevant_document_content = ensemble_response.get('relevant_document_content', None)
            if relevant_document_content:
                self.log("Planning Agent received relevant uploaded document content from Ensemble Agent")
                response['relevant_document_content'] = relevant_document_content

            mcp_response = ensemble_response.get('mcp_response', None)
            if mcp_response:
                self.log("Planning Agent received response from MCP server")
                response['mcp_response'] = mcp_response

        # Generate natural response using LLM
        natural_response = self.synthesize_response(
            user_query,
            response.get('relevant_tickets', []),
            response['mcp_response'],
            response['recently_asked_questions'],
            response['relevant_document_content'] if 'relevant_document_content' in response else None,
            chat_history=chat_history
        )
        response['natural_response'] = natural_response

        return response