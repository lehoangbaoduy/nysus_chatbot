import datetime
from pydantic import BaseModel
from typing import List, Dict, Self
import re
from tqdm import tqdm
import time

class Company(BaseModel):
    """
    A class to Represent a company with a summary description
    """
    company_description: str
    location: str
    url: str

class Ticket(BaseModel):
    """
    A class to represent a possible ticket: a related info where we think
    it should be related to the user question
    """
    company: Company
    ticket_number: int
    created_at: datetime.datetime
    updated_at: datetime.datetime
    url: str
    author: str
    subject: str
    description: str
    score: float

class TicketSelection(BaseModel):
    """
    A class to Represent a list of selected tickets
    """
    tickets: List[Ticket]

class Question(BaseModel):
    """
    A class to represent a recently asked question: a cached info where we store
    recently asked questions previously asked by users
    """
    question: str
    answer: str
    database: dict
    resolution: bool

class ScannerAgentResponse(BaseModel):
    """
    A class to Represent a list of cached recently asked questions
    """
    questions: List[Question]
    pdf_context: str = None  # Context extracted from uploaded PDF files