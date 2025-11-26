# imports
import re
import datetime
import sys
from typing import List, Dict
from agents.tickets import Ticket, Company, TicketSelection
# OpenAI imports - keeping embeddings for vectorstore compatibility
from langchain_openai import OpenAIEmbeddings
# Note: Ollama embeddings have different dimensions than OpenAI (768 vs 1536)
# To use Ollama embeddings, you'd need to recreate the vectorstore with:
# from langchain_ollama import OllamaEmbeddings
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

        # To use Ollama embeddings (dimension 768), first recreate the vectorstore:
        # self.embeddings = OllamaEmbeddings(model="nomic-embed-text")

        self.vectorstore = Chroma(
            persist_directory=self.DB_NAME,
            embedding_function=self.embeddings
        )
        self.collection = collection  # Keep for backward compatibility
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