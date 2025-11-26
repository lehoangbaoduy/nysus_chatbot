from typing import Optional, List, Dict
from agents.agent import Agent
from agents.tickets import Company, Ticket
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

    def synthesize_response(self, user_query: str, rag_response: Optional[str], mcp_response: Optional[Dict], recently_asked_questions: List[str], chat_history: List = []) -> str:
        """
        Use LLM to generate a natural, conversational response based on retrieved information
        :param user_query: The user's original question
        :param rag_response: Natural language response from RAG system (Frontier Agent)
        :param mcp_response: Response from MCP database query
        :param chat_history: Previous conversation context
        :return: Natural language response string
        """
        self.log("Synthesizing natural response using LLM")

        # Build context from recently asked questions
        questions_context = ""
        if recently_asked_questions:
            questions_context = "\n\nRecently asked similar questions:\n"
            for i, question in enumerate(recently_asked_questions, 1):
                if hasattr(question, 'question'):
                    questions_context += f"\n{i}. {question.question}\n"
                else:
                    questions_context += f"\n{i}. {question}\n"

        # Build context from RAG response
        rag_context = ""
        if rag_response:
            rag_context = f"\n\nRAG System Response (from ticket knowledge base):\n{rag_response}\n"

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
        Your role is to provide clear, conversational answers based on RAG responses from ticket knowledge base, database information, and cached questions.

        Guidelines:
        - Be conversational and natural, not robotic
        - If recently asked similar questions are provided, acknowledge them and reference relevant ones
        - If a RAG system response is provided, DO NOT mention database info from the RAG response, ONLY include any info regards related ticket,
        - Summarize technical information in an accessible way.
        - If database results are provided, inintegrate them naturally into your response
        - If both RAG and database info are provided, USE THE DATABASE CONTEXT FOR SQL QUERY DETAILS.
        - Stay focused on the user's question
        - If information is incomplete, acknowledge it naturally
        - Use a friendly, professional tone
        """

        user_prompt = f"""User Question: {user_query}
        {history_context}
        {questions_context}
        {rag_context}
        {db_context}

        Please provide a natural, helpful response to the user's question based on the information above.

        IMPORTANT:
        - If a RAG response is provided, use it as the main source for ticket information
        - Synthesize the information into a coherent, conversational answer
        - Don't just repeat the RAG response - enhance it with database results if available
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
            if recently_asked_questions:
                fallback += f"Found {len(recently_asked_questions)} similar question(s) asked recently.\n"
            if rag_response:
                fallback += f"RAG Response: {rag_response}\n\n"
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
        selection = self.scanner.scan(memory=memory, user_query=user_query, uploaded_files=uploaded_files)

        if selection and selection.questions:
            self.log(f"Planning Agent received {len(selection.questions)} questions from Scanner")
            print(f"Found {len(selection.questions)} relevant cached questions:")
            for i, question in enumerate(selection.questions, 1):
                print(f"  {i}. Question: {question.question}")

            self.log("Planning Agent has completed a run")
            response['recently_asked_questions'] = selection.questions
        else:
            self.log("No relevant recently asked questions found. Asking Ensemble Agent to search knowledge base.")

            # Use EnsembleAgent to answer the question by searching ChromaDB and potentially MCP server
            self.log("Planning Agent is delegating to Ensemble Agent to answer the user question")
            ensemble_response = self.ensemble.answer_question(user_query, chat_history, n_results=5, schema=schema)

            relevant_tickets_response = ensemble_response.get('relevant_tickets_response', None)
            if relevant_tickets_response:
                # relevant_tickets_response is a tuple: (answer_text, chunks)
                rag_answer, rag_chunks = relevant_tickets_response
                self.log(f"Planning Agent received RAG response from knowledge base with {len(rag_chunks)} relevant chunks")
                response['rag_response'] = rag_answer
                response['rag_chunks'] = rag_chunks

            mcp_response = ensemble_response.get('mcp_response', None)
            if mcp_response:
                self.log("Planning Agent received response from MCP server")
                response['mcp_response'] = mcp_response

        # Generate natural response using LLM
        natural_response = self.synthesize_response(user_query, response['rag_response'], response['mcp_response'], response['recently_asked_questions'], chat_history)
        response['natural_response'] = natural_response

        return response