# imports
import re
import datetime
# OpenAI imports - keeping embeddings for vectorstore compatibility
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
# Note: Ollama embeddings have different dimensions than OpenAI (768 vs 1536)
# To use Ollama embeddings, you'd need to recreate the vectorstore with:
# from langchain_ollama import OllamaEmbeddings
from typing import List
from agents.base_classes import Ticket, Company
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from PyPDF2 import PdfReader
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
                - Similarity Score, it is the actual score from the similarity_search_with_score. ALWAYS include this score (how relevant the ticket is to the question)
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

    def get_pdf_text(self, pdf_docs: List) -> str:
        """
        Extract text from uploaded PDF files
        :param pdf_docs: List of PDF file objects
        :return: Combined text from all PDFs
        """
        text = ""
        for pdf in pdf_docs:
            try:
                pdf_reader = PdfReader(pdf)
                for page in pdf_reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted
            except Exception as e:
                self.log(f"Error reading PDF: {e}")
        return text

    def get_text_chunks(self, text: str) -> List[str]:
        """
        Split text into chunks for embedding
        :param text: Raw text to split
        :return: List of text chunks
        """
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        return chunks

    def get_vectorstore(self, text_chunks: List[str]):
        """
        Create FAISS vectorstore from text chunks
        :param text_chunks: List of text chunks
        :return: FAISS vectorstore
        """
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=self.embeddings)
        return vectorstore

    def query_vectorstore(self, vectorstore, query: str, k: int = 5) -> List[str]:
        """
        Query the vectorstore for relevant documents
        :param vectorstore: FAISS vectorstore
        :param query: User query
        :param k: Number of results to return
        :return: List of relevant document chunks
        """
        try:
            docs = vectorstore.similarity_search_with_score(query, k=k)
            return [doc.page_content for doc, score in docs]
        except Exception as e:
            self.log(f"Error querying vectorstore: {e}")
            return []

    def invoke_LLM_for_uploaded_content(self, user_query, pdf_context) -> str:
        """
        Create a user prompt for OpenAI based on the matching uploaded file content
        and call LLM to extract relevant information
        """
        system_prompt = """
        You are a helpful assistant analyzing PDF documents to answer user questions.
        You will be provided with relevant excerpts from PDF documents that were uploaded by the user.

        Your task is to:
        1. Carefully analyze the provided PDF content
        2. Extract information that is relevant to the user's question
        3. Provide a clear, accurate, and comprehensive answer based on the PDF content
        4. If the PDF content doesn't contain relevant information to answer the question, say so
        5. Cite specific details from the PDF when possible

        Be accurate and only use information present in the provided PDF excerpts.
        """

        user_prompt = f"""
        Based on the following excerpts from the uploaded PDF document(s):

        ---PDF CONTENT---
        {pdf_context}
        ---END PDF CONTENT---

        Please answer this question: {user_query}
        """

        try:
            self.log("Calling LLM to analyze PDF content")
            response = self.client.chat.completions.create(
                model=self.MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )

            result = response.choices[0].message.content
            self.log(f"LLM analysis completed: {len(result)} characters")
            return result

        except Exception as e:
            self.log(f"Error calling LLM for PDF analysis: {e}")
            return f"Error analyzing PDF content: {str(e)}"


    def find_relevant_uploaded_content(self, user_query: str, uploaded_files: List = []) -> str:
        # Process uploaded PDF files if any
        pdf_context = None
        if uploaded_files:
            self.log(f"Processing {len(uploaded_files)} uploaded PDF file(s)")
            self.log(f"Uploaded files: {[f.name for f in uploaded_files]}")
            try:
                # Extract text from PDFs
                raw_text = self.get_pdf_text(uploaded_files)

                if raw_text.strip():
                    self.log(f"Extracted {len(raw_text)} characters from PDFs")

                    # Get text chunks
                    text_chunks = self.get_text_chunks(raw_text)
                    self.log(f"Split text into {len(text_chunks)} chunks")

                    # Create vectorstore
                    vectorstore = self.get_vectorstore(text_chunks)
                    self.log("Created vectorstore from PDF content")

                    # Query vectorstore for relevant content
                    relevant_docs = self.query_vectorstore(vectorstore, user_query, k=3)
                    self.log(f"Retrieved {len(relevant_docs)} relevant chunks from PDFs")

                    # Combine relevant chunks into context
                    if relevant_docs:
                        pdf_context = "\n\n".join(relevant_docs)
                else:
                    self.log("No text could be extracted from PDFs")
            except Exception as e:
                self.log(f"Error processing PDFs: {e}")

            result = self.invoke_LLM_for_uploaded_content(user_query, pdf_context)
            self.log("Frontier Agent completed analysis of uploaded PDF content")
            return result

