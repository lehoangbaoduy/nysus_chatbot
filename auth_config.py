"""
Google OAuth Configuration for Nysus Chatbot
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Google OAuth Configuration
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
REDIRECT_URI = os.getenv("REDIRECT_URI", "http://localhost:8501")

# Allowed email domains for authentication
ALLOWED_DOMAINS = ["nysus.net", "nysus.com"]

# Session configuration
SESSION_COOKIE_NAME = "nysus_chatbot_session"
SESSION_MAX_AGE = 86400  # 24 hours in seconds
