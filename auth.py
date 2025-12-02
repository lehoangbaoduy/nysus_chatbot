"""
Google OAuth Authentication Module for Streamlit
"""
import streamlit as st
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
import os
import json
from auth_config import GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, REDIRECT_URI, ALLOWED_DOMAINS

# OAuth 2.0 scopes for Google authentication
SCOPES = ['openid', 'https://www.googleapis.com/auth/userinfo.email', 'https://www.googleapis.com/auth/userinfo.profile']

def init_oauth_flow():
    """Initialize Google OAuth flow"""
    client_config = {
        "web": {
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [REDIRECT_URI],
        }
    }

    flow = Flow.from_client_config(
        client_config,
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI
    )
    return flow

def get_authorization_url():
    """Get Google OAuth authorization URL"""
    flow = init_oauth_flow()
    authorization_url, state = flow.authorization_url(
        access_type='offline',
        include_granted_scopes='true',
        prompt='select_account'
    )
    return authorization_url, state

def get_user_info(credentials):
    """Get user information from Google"""
    try:
        service = build('oauth2', 'v2', credentials=credentials)
        user_info = service.userinfo().get().execute()
        return user_info
    except Exception as e:
        st.error(f"Error fetching user info: {str(e)}")
        return None

def exchange_code_for_token(code, state):
    """Exchange authorization code for access token"""
    try:
        flow = init_oauth_flow()
        flow.fetch_token(code=code)
        credentials = flow.credentials
        return credentials
    except Exception as e:
        st.error(f"Error exchanging code for token: {str(e)}")
        return None

def is_email_allowed(email):
    """Check if email domain is in allowed list"""
    if not email:
        return False
    domain = email.split('@')[-1].lower()
    return domain in ALLOWED_DOMAINS

def check_authentication():
    """
    Check if user is authenticated and from allowed domain.
    Returns True if authenticated, False otherwise.
    """
    # Initialize session state for authentication
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user_info' not in st.session_state:
        st.session_state.user_info = None
    if 'access_denied' not in st.session_state:
        st.session_state.access_denied = False

    # Check if credentials are missing
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        st.error("⚠️ Google OAuth credentials not configured. Please set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET in your .env file.")
        st.stop()

    # Handle OAuth callback
    query_params = st.query_params
    if 'code' in query_params and not st.session_state.authenticated:
        code = query_params['code']
        state = query_params.get('state', '')

        # Exchange code for credentials
        credentials = exchange_code_for_token(code, state)

        if credentials:
            user_info = get_user_info(credentials)

            if user_info:
                email = user_info.get('email', '')

                # Check if email domain is allowed
                if is_email_allowed(email):
                    st.session_state.authenticated = True
                    st.session_state.user_info = user_info
                    st.session_state.access_denied = False
                    # Clear query parameters
                    st.query_params.clear()
                    st.rerun()
                else:
                    # Mark access as denied and clear query parameters
                    st.session_state.access_denied = True
                    st.query_params.clear()
                    st.rerun()

    return st.session_state.authenticated

def show_login_page():
    """Display the login page"""
    st.markdown("""
        <style>
        .login-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            padding: 1rem 0;
            margin: 0 0 1rem 0;
        }
        .login-title {
            font-size: 2rem;
            font-weight: bold;
            margin-bottom: 0.3rem;
            background: linear-gradient(45deg, #2196F3, #21CBF3);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .login-subtitle {
            font-size: 1rem;
            color: #666;
            margin-bottom: 0.3rem;
        }
        .login-info {
            font-size: 0.85rem;
            color: #999;
            margin-top: 0.5rem;
        }
        .login-info p {
            margin: 0.2rem 0;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="login-container">
            <div class="login-title">🤖 NAAS - Nysus Automated Assistant for MES</div>
            <div class="login-subtitle">AI-Powered Support Assistant</div>
        </div>
    """, unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # Show access denied message if applicable
        if st.session_state.get('access_denied', False):
            st.error("❌ Access Denied: Only @nysus.net and @nysus.com email addresses are allowed.")
            st.warning("Please sign in with an authorized Nysus email address.")
        else:
            st.info("🔐 Please sign in with your Nysus email address")

        if st.button("Sign in with Google", use_container_width=True, type="secondary", icon="🔑"):
            # Clear access denied flag when retrying
            st.session_state.access_denied = False
            authorization_url, state = get_authorization_url()
            st.session_state.oauth_state = state
            # Redirect to Google OAuth
            st.markdown(f'<meta http-equiv="refresh" content="0;url={authorization_url}">', unsafe_allow_html=True)
            st.markdown(f'<meta http-equiv="refresh" content="0;url={authorization_url}">', unsafe_allow_html=True)

        st.markdown("""
            <div class="login-info">
                <p>✓ Secure authentication via Google</p>
                <p>✓ Access restricted to Nysus domain emails</p>
            </div>
        """, unsafe_allow_html=True)

def show_user_info():
    """Display user information in sidebar"""
    if st.session_state.authenticated and st.session_state.user_info:
        user_info = st.session_state.user_info
        with st.sidebar:
            st.markdown("### 🔐 User Info")

            # Display user profile picture and info side by side
            if 'picture' in user_info:
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.image(user_info['picture'], width=50)
                with col2:
                    st.markdown(f"<p style='margin-bottom: 0;'><strong>ⓘ {user_info.get('name', 'User')}</strong></p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='margin-top: 0; font-size: 0.85em;'>✉︎ {user_info.get('email', '')}</p>", unsafe_allow_html=True)
            else:
                st.markdown(f"<p style='margin-bottom: 0;'><strong>ⓘ {user_info.get('name', 'User')}</strong></p>", unsafe_allow_html=True)
                st.markdown(f"<p style='margin-top: 0; font-size: 0.85em;'>✉︎ {user_info.get('email', '')}</p>", unsafe_allow_html=True)

            st.markdown("")  # Small spacing
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🔑 Logout", use_container_width=True):
                    logout()

def logout():
    """Log out the user"""
    st.session_state.authenticated = False
    st.session_state.user_info = None
    if 'oauth_state' in st.session_state:
        del st.session_state.oauth_state
    st.rerun()
