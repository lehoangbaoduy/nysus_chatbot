import os
import sys
import logging
import json
from typing import List, Optional
from dotenv import load_dotenv
import chromadb
from agents.planning_agent import PlanningAgent
from agents.tickets import Ticket
from sklearn.manifold import TSNE
import numpy as np


# Colors for logging
BG_BLUE = '\033[44m'
WHITE = '\033[37m'
RESET = '\033[0m'

# Colors for plot
CATEGORIES = ['Appliances', 'Automotive', 'Cell_Phones_and_Accessories', 'Electronics','Musical_Instruments', 'Office_Products', 'Tools_and_Home_Improvement', 'Toys_and_Games']
COLORS = ['red', 'blue', 'brown', 'orange', 'yellow', 'green' , 'purple', 'cyan']

def init_logging():
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s] [Agents] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S %z",
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)

class ChatbotAgentFramework:

    DB = "data/documents/os_tickets_vectorstore"
    MEMORY_FILENAME = "memory.json"

    def __init__(self, db_connection_params=None):
        init_logging()
        load_dotenv()
        # ChromaDB collection is no longer needed - agents will use LangChain directly
        self.memory = self.read_memory()
        self.planner = None
        self.db_connection_params = db_connection_params
        self.log(f"ChatbotAgentFramework initialized with db_connection_params: {db_connection_params is not None}")

    def init_agents_as_needed(self):
        if not self.planner:
            self.log("Initializing Agent Framework")
            self.log(f"Passing db_connection_params to PlanningAgent: {self.db_connection_params is not None}")
            self.planner = PlanningAgent(db_connection_params=self.db_connection_params)
            self.log("Chatbot Agent Framework is ready")

    def update_db_connection_params(self, db_connection_params):
        """Update database connection parameters in the framework and all agents"""
        self.log(f"Updating db_connection_params: {db_connection_params is not None}")
        self.db_connection_params = db_connection_params

        # If agents are already initialized, update their connection params
        if self.planner and hasattr(self.planner, 'ensemble') and hasattr(self.planner.ensemble, 'mcp'):
            new_params = {
                'host': db_connection_params['host'],
                'port': db_connection_params['port'],
                'user': db_connection_params['user'],
                'password': db_connection_params['password']
            }
            current_params = self.planner.ensemble.mcp.db_connection_params

            # Only update if params actually changed
            if current_params != new_params:
                self.log("Connection params changed, updating MCP Agent")
                self.planner.ensemble.mcp.set_connection_params(
                    db_connection_params['host'],
                    db_connection_params['port'],
                    db_connection_params['user'],
                    db_connection_params['password'],
                    db_connection_params.get('driver', 'ODBC Driver 17 for SQL Server')
                )
            else:
                self.log("Connection params unchanged, skipping update")
        else:
            # Agents not initialized yet - initialize them now with the new params
            self.log("Agents not yet initialized, initializing with new connection params")
            self.init_agents_as_needed()

    def read_memory(self) -> List[Ticket]:
        if os.path.exists(self.MEMORY_FILENAME):
            try:
                with open(self.MEMORY_FILENAME, "r") as file:
                    content = file.read().strip()
                    if not content:
                        self.log("Memory file is empty, initializing with empty list")
                        return []
                    data = json.loads(content)
                opportunities = [Ticket(**item) for item in data]
                return opportunities
            except json.JSONDecodeError as e:
                self.log(f"Error parsing memory file: {e}. Initializing with empty list")
                return []
        return []

    def write_memory(self) -> None:
        data = [Ticket.dict() for Ticket in self.memory]
        with open(self.MEMORY_FILENAME, "w") as file:
            json.dump(data, file, indent=2)

    def log(self, message: str):
        text = BG_BLUE + WHITE + "[Agent Framework] " + message + RESET
        logging.info(text)

    def run(self, user_query, chat_history, uploaded_files, schema=None):
        # Note: init_agents_as_needed() already called in get_agent_framework()
        logging.info("Kicking off Planning Agent")
        result = self.planner.plan(memory=self.memory, user_query=user_query, chat_history=chat_history, uploaded_files=uploaded_files, schema=schema)
        logging.info(f"Planning Agent has completed and returned: {type(result)}")
        # Return result directly - don't append to memory as it causes duplicate processing
        return result

    @classmethod
    def get_plot_data(cls, max_datapoints=10000):
        client = chromadb.PersistentClient(path=cls.DB)
        collection = client.get_or_create_collection('os_tickets')
        result = collection.get(include=['embeddings', 'documents', 'metadatas'], limit=max_datapoints)
        vectors = np.array(result['embeddings'])
        documents = result['documents']
        categories = [metadata['category'] for metadata in result['metadatas']]
        colors = [COLORS[CATEGORIES.index(c)] for c in categories]
        tsne = TSNE(n_components=3, random_state=42, n_jobs=-1)
        reduced_vectors = tsne.fit_transform(vectors)
        return documents, reduced_vectors, colors


if __name__=="__main__":
    ChatbotAgentFramework().run()
