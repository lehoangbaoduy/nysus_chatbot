# imports
import re
import datetime
import sys
from typing import List, Dict
from agents.tickets import Ticket, Company

# OpenAI imports - keeping embeddings for vectorstore compatibility
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
# Note: Ollama embeddings have different dimensions than OpenAI (768 vs 1536)
# To use Ollama embeddings, you'd need to recreate the vectorstore with:
# from langchain_ollama import OllamaEmbeddings
from pydantic import BaseModel, Field
from langchain_chroma import Chroma
from agents.agent import Agent


class FrontierAgent(Agent):

    name = "Frontier Agent"
    color = Agent.BLUE

    # No LLM calls in this agent, only embeddings for search
    DB_NAME = "data/documents/os_tickets_vectorstore/frontier"

    def __init__(self, collection):
        """
        Set up this instance by connecting to the Chroma Datastore with embeddings.
        NOTE: Using OpenAI embeddings because the vectorstore was created with them.
        This is a minimal cost operation (only for similarity search, not generation).

        To switch to Ollama embeddings:
        1. Uncomment: from langchain_ollama import OllamaEmbeddings
        2. Replace: self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        3. Re-run: data_parsing_scripts/initialize_vectorstore.py with Ollama embeddings
        """
        self.log("Initializing Frontier Agent")
        self.log("Frontier Agent is setting up with OpenAI embeddings (for vectorstore compatibility)")

        # Initialize embeddings and vectorstore for RAG
        # IMPORTANT: Using OpenAI embeddings because vectorstore was created with them (dimension 1536)
        self.embeddings = OpenAIEmbeddings()
        self.client = OpenAI()
        self.embedding_model = "text-embedding-3-large"
        self.MODEL = "gpt-4o-mini"

        # To use Ollama embeddings (dimension 768), first recreate the vectorstore:
        # self.embeddings = OllamaEmbeddings(model="nomic-embed-text")

        self.vectorstore = Chroma(
            persist_directory=self.DB_NAME,
            embedding_function=self.embeddings
        )
        self.collection = collection  # Keep for backward compatibility

        self.RETRIEVAL_K = 20
        self.FINAL_K = 10

        self.SYSTEM_PROMPT = """
        You are a knowledgeable, friendly MES support analyst to retrieve related support tickets for to solve user issues.
        You should use the provided extracts from the Knowledge Base to answer the user's question.

        IMPORTANT OUTPUT RULE:
            1. Try your best to include these infos when output tickets:
                - Ticket Number
                - Company Name
                - Subject / Topic
                - Author
                - Created and Updated Dates
                - URL to the ticket
                - Full Description of the ticket including all interactions (requests, notes, responses)
            2. DO NOT make up any SQL queries or database info.
            3. ONLY use the provided extracts from the Knowledge Base to answer the question.
            4. Be accurate, relevant and complete in your answers.
            5. If you don't know the answer, say so.

        Your answer will be evaluated for accuracy, relevance and completeness, so make sure it only answers the question and fully answers it.
        For context, here are specific extracts from the Knowledge Base that might be directly relevant to the user's question:
        {context}

        With this context, please answer the user's question. Be accurate, relevant and complete.
        """
        self.log("Frontier Agent is ready")

    def find_relevant_tickets(self, user_query: str, n_results: int = 5) -> List[Ticket]:
        """
        Search for relevant tickets from ChromaDB based on user query
        Uses LangChain's similarity search with OpenAI embeddings (matching the vectorstore)
        NOTE: This uses OpenAI embeddings for search only - minimal cost operation
        :param user_query: the user's question
        :param n_results: number of results to return
        :return: a list of Ticket objects following the TicketSelection schema
        """
        self.log(f"Frontier Agent is searching ChromaDB for tickets relevant to: '{user_query}'")

        # Use LangChain's similarity search (uses OpenAI embeddings to match vectorstore dimension)
        results = self.vectorstore.similarity_search_with_score(user_query, k=n_results)

        self.log(f"Frontier Agent found {len(results)} relevant tickets")

        # Convert to Ticket objects following the schema
        tickets = []
        for doc, score in results:
            # Extract ticket information from the document
            content = doc.page_content
            metadata = doc.metadata            # Parse ticket number from markdown header like "## Ticket #029702 (ID: 29884)"
            ticket_match = re.search(r'## Ticket #(\d+)', content)
            ticket_number = int(ticket_match.group(1)) if ticket_match else 0

            # Parse company from markdown
            company_match = re.search(r'\*\*Company:\*\* (.+)', content)
            company_name = company_match.group(1).strip() if company_match else ""

            # Parse subject/topic
            topic_match = re.search(r'\*\*Topic:\*\* (.+)', content)
            subject = topic_match.group(1).strip() if topic_match else ""

            # Parse dates
            created_match = re.search(r'\*\*Created:\*\* (.+?) \|', content)
            created_str = created_match.group(1).strip() if created_match else ""

            updated_match = re.search(r'\*\*Updated:\*\* (.+)', content)
            updated_str = updated_match.group(1).strip() if updated_match else ""

            # Parse datetime strings
            try:
                created_at = datetime.datetime.strptime(created_str, "%Y-%m-%d %H:%M:%S") if created_str else datetime.datetime.now()
            except:
                created_at = datetime.datetime.now()

            try:
                updated_at = datetime.datetime.strptime(updated_str, "%Y-%m-%d %H:%M:%S") if updated_str else datetime.datetime.now()
            except:
                updated_at = datetime.datetime.now()

            # Parse URL
            url_match = re.search(r'\*\*URL:\*\* (.+)', content)
            url = url_match.group(1).strip() if url_match else ""

            # Parse author from response, note, or initial request (appears after Date in subsections)
            # Look for pattern: **Date:** ... | **Author:** Name (email) or just **Author:** Name (email)
            author = ""
            # Try multiline pattern first to capture author after Date
            author_match = re.search(r'\*\*Date:\*\*[^\n]+\|\s*\*\*Author:\*\*\s*([^\n]+)', content, re.MULTILINE)
            if author_match:
                author = author_match.group(1).strip()
            else:
                # Try simpler pattern - just find any **Author:** line
                author_match = re.search(r'\*\*Author:\*\*\s*([^\n]+)', content, re.MULTILINE)
                if author_match:
                    author = author_match.group(1).strip()

            # If no author found and we have a source file, try to get more context from metadata
            if not author and 'source' in metadata:
                # The chunk might not contain author info - it could be in another chunk
                # For now, leave it empty - we could optionally load the full file later
                pass

            # Clean up author string (remove extra whitespace)
            author = ' '.join(author.split()) if author else ""            # Create Company object
            company = Company(
                company_description=company_name,
                location="",  # Not available in current format
                url=""  # Company URL not available
            )

            # Create Ticket object following the schema
            ticket = Ticket(
                company=company,
                ticket_number=ticket_number,
                created_at=created_at,
                updated_at=updated_at,
                url=url,
                author=author,
                subject=subject if subject else company_name,
                description=content,
                score=score
            )

            tickets.append(ticket)

        return tickets

    def rerank(self, question, chunks):
        system_prompt = """
            You are a document re-ranker.
            You are provided with a question and a list of relevant chunks of text from a query of a knowledge base.
            The chunks are provided in the order they were retrieved; this should be approximately ordered by relevance, but you may be able to improve on that.
            You must rank order the provided chunks by relevance to the question, with the most relevant chunk first.
            Reply only with the list of ranked chunk ids, nothing else. Include all the chunk ids you are provided with, reranked.
            """

        user_prompt = f"The user has asked the following question:\n\n{question}\n\nOrder all the chunks of text by relevance to the question, from most relevant to least relevant. Include all the chunk ids you are provided with, reranked.\n\n"
        user_prompt += "Here are the chunks:\n\n"
        for index, chunk in enumerate(chunks):
            user_prompt += f"# CHUNK ID: {index + 1}:\n\n{chunk.page_content}\n\n"
        user_prompt += "Reply only with the list of ranked chunk ids, nothing else."
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        class RankOrder(BaseModel):
            order: list[int] = Field(
                description="The order of relevance of chunks, from most relevant to least relevant, by chunk id number"
            )

        response = self.client.chat.completions.parse(model=self.MODEL, messages=messages, response_format=RankOrder)
        reply = response.choices[0].message.content
        order = RankOrder.model_validate_json(reply).order

        # Validate indices: filter out invalid indices and handle edge cases
        valid_indices = []
        chunks_len = len(chunks)
        for i in order:
            # Check if index is within valid range (1-based indexing)
            if 1 <= i <= chunks_len and (i - 1) not in valid_indices:
                valid_indices.append(i - 1)

        # If no valid indices or missing chunks, return original order
        if not valid_indices:
            self.log(f"Warning: No valid reranked indices returned, using original order")
            return chunks

        # Add any missing chunks at the end
        result = [chunks[idx] for idx in valid_indices]
        for idx in range(chunks_len):
            if idx not in valid_indices:
                result.append(chunks[idx])

        return result

    def make_rag_messages(self, question, history, chunks):
        context = "\n\n".join(
            f"Extract from {chunk.metadata['source']}:\n{chunk.page_content}" for chunk in chunks
        )
        system_prompt = self.SYSTEM_PROMPT.format(context=context)

        # Validate and filter history messages to ensure they have required fields
        valid_history = []
        for msg in history:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                # Only include valid roles
                if msg["role"] in ["user", "assistant", "system"]:
                    valid_history.append({"role": msg["role"], "content": msg["content"]})

        return (
            [{"role": "system", "content": system_prompt}]
            + valid_history
            + [{"role": "user", "content": question}]
        )

    def merge_chunks(self, chunks, reranked):
        merged = chunks[:]
        existing = [chunk.page_content for chunk in chunks]
        for chunk in reranked:
            if chunk.page_content not in existing:
                merged.append(chunk)
        return merged

    def fetch_context_unranked(self, question):
        # Use LangChain's vectorstore similarity search instead of raw ChromaDB query
        # This properly handles embeddings and returns results in the expected format
        results = self.vectorstore.similarity_search_with_score(question, k=self.RETRIEVAL_K)
        chunks = []

        class Result(BaseModel):
            page_content: str
            metadata: dict

        for doc, score in results:
            chunks.append(Result(page_content=doc.page_content, metadata=doc.metadata))
        return chunks

    def rewrite_query(self, user_query, history=[]):
        """Rewrite the user's query to be a more specific question that is more likely to surface relevant content in the Knowledge Base."""
        message = f"""
            You are in a conversation with a user, answering questions of the user to get related tickets.
            You are about to look up information in a Knowledge Base to answer the user's question.

            This is the history of your conversation so far with the user:
            {history}

            And this is the user's current question:
            {user_query}

            Respond only with a short, refined question that you will use to search the Knowledge Base.
            It should be a short specific question most likely to surface content. Focus on the question details.
            IMPORTANT: Respond ONLY with the precise knowledgebase query, nothing else.
            """
        response = self.client.chat.completions.parse(model=self.MODEL, messages=[{"role": "system", "content": message}])
        return response.choices[0].message.content

    def fetch_context(self, user_query: str) -> list:
        rewritten_question = self.rewrite_query(user_query)
        chunks1 = self.fetch_context_unranked(user_query)
        chunks2 = self.fetch_context_unranked(rewritten_question)
        chunks = self.merge_chunks(chunks1, chunks2)
        reranked = self.rerank(user_query, chunks)
        return reranked[:self.FINAL_K]

    def answer_question_with_rag(self, user_query: str, chat_history: list[dict] = []) -> tuple[str, list]:
        """
        Answer a question using RAG and return the answer and the retrieved context
        """
        self.log(f"Frontier Agent is searching ChromaDB for tickets relevant to: '{user_query}'")
        chunks = self.fetch_context(user_query)
        self.log(f"Frontier Agent found {len(chunks)} chunks of vectors that could contain relevant tickets")
        messages = self.make_rag_messages(user_query, chat_history, chunks)
        response = self.client.chat.completions.parse(model=self.MODEL, messages=messages)
        return response.choices[0].message.content, chunks
