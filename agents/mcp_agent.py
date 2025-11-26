import pyodbc
import sys
import json
import re
import urllib
from langchain_community.utilities import SQLDatabase
from typing import Dict, List, Optional
# OpenAI import (commented out for cost savings during testing)
from openai import OpenAI
# from langchain_ollama import ChatOllama
from agents.agent import Agent


class MCPAgent(Agent):
    """
    MCP (Model Context Protocol) Agent for querying SQL databases
    Handles database schema retrieval and natural language to SQL conversion
    """

    name = "MCP Agent"
    color = Agent.MAGENTA

    # OpenAI model (commented out)
    MODEL = "gpt-4o-mini"
    # Ollama model (using for testing)
    # MODEL = "llama3.2"  # or "mistral", "qwen2.5", etc.

    def __init__(self, db_connection_params: Optional[Dict[str, str]] = None):
        """
        Initialize MCP Agent with database connection parameters
        :param db_connection_params: dict with keys: host, port, user, password, driver
        """
        self.log("Initializing MCP Agent")
        # OpenAI initialization (commented out for cost savings)
        self.client = OpenAI()
        # Ollama initialization
        # self.llm = ChatOllama(model=self.MODEL, temperature=0, format="json")
        self.db_connection_params = db_connection_params or {}
        self.schema_cache = None
        self.log(f"MCP Agent is ready with db_connection_params: {self.db_connection_params is not None and bool(self.db_connection_params)}")
        if self.db_connection_params:
            self.log(f"Connection params keys: {list(self.db_connection_params.keys())}")

    def set_connection_params(self, host: str, port: str, user: str, password: str, driver: str = "ODBC Driver 17 for SQL Server"):
        """
        Set or update database connection parameters
        """
        self.log(f"Updating connection params - host: {host}, user: {user}")
        self.db_connection_params = {
            'host': host,
            'port': port,
            'user': user,
            'password': password,
            'driver': driver
        }
        self.schema_cache = None  # Clear cache when connection changes
        self.log("Database connection parameters updated")

    def get_all_databases(self, db_connection_params):
        user = db_connection_params.get('user')
        password = db_connection_params.get('password')
        host = db_connection_params.get('host')
        port = db_connection_params.get('port')
        driver = db_connection_params.get('driver', "ODBC Driver 17 for SQL Server")

        odbc_str = (
            f"DRIVER={{{driver}}};"
            f"SERVER={host};"
            f"UID={user};"
            f"PWD={password};"
            "Trusted_Connection=no;"
        )
        conn = pyodbc.connect(odbc_str)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT name FROM sys.databases
            WHERE name NOT IN ('master', 'tempdb', 'model', 'msdb');
        """)
        databases = [row[0] for row in cursor.fetchall()]
        conn.close()

        db_map = {}
        for db_name in databases:
            odbc_db_str = (
                f"DRIVER={{{driver}}};"
                f"SERVER={host};"
                f"DATABASE={db_name};"
                f"UID={user};"
                f"PWD={password};"
                "Trusted_Connection=no;"
            )
            params = urllib.parse.quote_plus(odbc_db_str)
            db_uri = f"mssql+pyodbc:///?odbc_connect={params}"
            try:
                db_map[db_name] = SQLDatabase.from_uri(db_uri)
            except Exception as e:
                print(f"⚠️ Could not connect to {db_name}: {e}")
        return db_map

    def smart_run_query(self, query: str, db_map: dict):
        query = query.strip()
        if not re.match(r"(?i)^(SELECT|EXEC|WITH)\b", query):
            raise ValueError(f"Unsafe SQL detected:\n{query}")
        db_pattern = re.compile(r"\bFROM\s+(\[?([A-Za-z0-9_]+)\]?\.)", re.IGNORECASE)
        match = db_pattern.search(query)
        if match:
            db_name = match.group(2)
            if db_name in db_map:
                return db_map[db_name].run(query)
            else:
                raise ValueError(f"❌ Could not find database '{db_name}' in the connected server.")
        else:
            for db_name, db_conn in db_map.items():
                try:
                    return db_conn.run(query)
                except Exception:
                    continue
            raise ValueError("❌ Query failed on all databases.")

    def get_schema(self) -> str:
        """
        Retrieve schema information from all databases on the SQL Server
        Returns a formatted string with database and table information
        """
        if self.schema_cache:
            self.log("Returning cached schema")
            return self.schema_cache

        # Check if connection params have actual values (not just empty dict)
        if not self.db_connection_params or not self.db_connection_params.get('host'):
            return "❌ No database connection parameters configured"

        self.log("Retrieving database schema from SQL Server")

        try:
            params = self.db_connection_params
            odbc_str = (
                f"DRIVER={{{params['driver']}}};"
                f"SERVER={params['host']};"
                f"UID={params['user']};"
                f"PWD={params['password']};"
                "Trusted_Connection=no;"
            )

            with pyodbc.connect(odbc_str) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT name FROM sys.databases
                    WHERE name NOT IN ('master', 'tempdb', 'model', 'msdb');
                """)
                databases = [row[0] for row in cursor.fetchall()]

                if not databases:
                    return "No user databases found on this SQL Server instance."

                schema_output = []
                for db_name in databases:
                    schema_output.append(f"\n=== Database: {db_name} ===")
                    try:
                        cursor.execute(f"USE [{db_name}]")
                        cursor.execute("""
                            SELECT TABLE_SCHEMA, TABLE_NAME
                            FROM INFORMATION_SCHEMA.TABLES
                            WHERE TABLE_TYPE = 'BASE TABLE'
                            ORDER BY TABLE_SCHEMA, TABLE_NAME;
                        """)
                        tables = cursor.fetchall()
                        if not tables:
                            schema_output.append("  (No tables found)")
                        else:
                            for schema, table in tables:
                                schema_output.append(f"  [{db_name}].{schema}.{table}")
                                # Get column information for each table
                                try:
                                    cursor.execute(f"""
                                        SELECT COLUMN_NAME, DATA_TYPE
                                        FROM [{db_name}].INFORMATION_SCHEMA.COLUMNS
                                        WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?
                                        ORDER BY ORDINAL_POSITION;
                                    """, schema, table)
                                    columns = cursor.fetchall()
                                    if columns:
                                        col_list = [f"{col[0]} ({col[1]})" for col in columns[:10]]  # Show first 10 columns
                                        schema_output.append(f"    Columns: {', '.join(col_list)}{'...' if len(columns) > 10 else ''}")
                                except Exception:
                                    pass  # Skip column details if unavailable
                    except Exception as e:
                        schema_output.append(f"  ⚠️ Could not read tables: {e}")

                self.schema_cache = "\n".join(schema_output)
                self.log(f"Successfully retrieved schema for {len(databases)} databases")
                return self.schema_cache

        except Exception as e:
            error_msg = f"❌ Failed to retrieve schema: {e}"
            self.log(error_msg)
            return error_msg

    def execute_sql_query(self, sql_query: str, database_name: str, db_map: dict) -> Dict:
        """
        Execute a SQL query on the specified database
        :param sql_query: The SQL query to execute
        :param database_name: The name of the database to query
        :return: Dict with results or error information
        """
        # Check if connection params have actual values (not just empty dict)
        if not self.db_connection_params or not self.db_connection_params.get('host'):
            return {"error": "No database connection parameters configured"}

        self.log(f"Executing SQL query on database: {database_name}")

        try:
            params = self.db_connection_params
            odbc_str = (
                f"DRIVER={{{params['driver']}}};"
                f"SERVER={params['host']};"
                f"UID={params['user']};"
                f"PWD={params['password']};"
                "Trusted_Connection=no;"
            )

            with pyodbc.connect(odbc_str) as conn:
                sql_query = sql_query.strip()
                if not re.match(r"(?i)^(SELECT|EXEC|WITH)\b", sql_query):
                    raise ValueError(f"Unsafe SQL detected:\n{sql_query}")

                db_pattern = re.compile(
                    r"\bFROM\s+\[?([A-Za-z0-9_]+)\]?\s*\.",
                    re.IGNORECASE
                )
                match = db_pattern.search(sql_query)
                if match:
                    db_name = match.group(1)
                    if db_name in db_map:
                        return db_map[db_name].run(sql_query)
                    else:
                        raise ValueError(f"❌ Could not find database '{db_name}' in the connected server.")
                else:
                    for db_name, db_conn in db_map.items():
                        try:
                            return db_conn.run(sql_query)
                        except Exception:
                            continue
                    raise ValueError("❌ Query failed on all databases.")

        except Exception as e:
            error_msg = f"Error executing query: {e}"
            self.log(error_msg)
            return {
                "success": False,
                "error": error_msg
            }

    def generate_sql_from_question(self, user_question: str, schema: str, chat_history: List = []) -> Dict:
        """
        Use OpenAI to convert natural language question to SQL query
        :param user_question: The user's question in natural language
        :param schema: The database schema information
        :return: Dict with generated SQL and target database
        """
        self.log("Generating SQL query from natural language question")

        system_prompt = """
            You are a Querymancer, a master database engineer with exceptional expertise in SQL.
            Your purpose is to transform natural language requests into precise, efficient SQL queries that deliver exactly what the user needs.

            CRITICAL RULES FOR QUERY ACCURACY:
            1. ONLY use column names that are explicitly provided in the database and table schema
            2. DO NOT assume or invent column names - if you're not certain a column exists, use SELECT * instead
            3. Each table belongs to EXACTLY ONE database - carefully identify which database contains the table you need
            4. NEVER query a table from the wrong database - verify the database name in the schema before writing your query
            5. ALWAYS use the FULL table path as shown in schema: [DatabaseName].schema.tablename
            6. If the user asks about specific fields that don't exist in the schema, explain what's missing
            7. If you feel the need to join between some tables across all databases, feel free to do so and make sure those tables exist in that database
            and mention your assumptions in the output if there's any.

            <instructions>
                <instruction>READ THE SCHEMA CAREFULLY - Each table is listed under its database with format [DATABASE].schema.table</instruction>
                <instruction>VERIFY the correct database for your target table - tables with similar names may exist in different databases</instruction>
                <instruction>NEVER assume column names - only use columns explicitly shown in the schema or use SELECT *</instruction>
                <instruction>Match column names EXACTLY as shown in the schema (case-sensitive)</instruction>
                <instruction>If columns are listed for a table, you MUST use those exact column names (do not invent alternatives)</instruction>
                <instruction>Use fully qualified table names [DATABASE].schema.table in your queries to avoid ambiguity</instruction>
                <instruction>If you're unsure about column names for a table, use SELECT * instead of guessing</instruction>
                <instruction>Double-check that the database name in your FROM clause matches the database where the table is actually located</instruction>
                <instruction>Take into account the chat conversation history for more info</instruction>
            </instructions>

            IMPORTANT OUTPUT FORMAT:
            1. Return your response in JSON format with these fields:
            - "database": The database name to query
            - "table": The table name to query
            - "sql_query": The SQL query to execute (use SELECT * if unsure about columns)
            - "explanation": Brief explanation of what the query does
            - "assumptions": List any assumptions made about column names (if applicable)

            2. Only generate SELECT queries for safety
            3. Use proper SQL Server syntax (square brackets for identifiers if needed)
            4. If the question cannot be answered with the available schema, set database to null
            5. Prefer SELECT * when column names are not explicitly provided in the schema

            Example response:
            {
                "database": "NYSUS_SEQUENCE_MES",
                "table": "modules",
                "sql_query": "SELECT TOP 10 * FROM NYSUS_SEQUENCE_MES.dbo.modules ORDER BY 1 DESC",
                "explanation": "Retrieves the first 10 modules from the table. Using SELECT * because specific column names were not provided in the schema.",
                "assumptions": []
            }
        """

        user_prompt = f"""Database Schema:
        {schema}
        Conversation History: {chat_history}
        User Question: {user_question}

        If you have to show the query, write only the SQL query between triple backticks.
        If you have to list out the databases or tables, list the full reference with this format DATABASE_NAME.dbo.TABLE_NAME, and follow this rule strictly.
        For example, 'NYSUS_SEQUENCE_MES.dbo.modules' or 'CUSTOM_YANFENG.dbo.messages'.
        Respond only in JSON format."""

        try:
            # OpenAI approach (commented out for cost savings)
            response = self.client.chat.completions.create(
                model=self.MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"}
            )
            result = json.loads(response.choices[0].message.content)

            # Ollama approach (for testing)
            # full_prompt = f"{system_prompt}\n\n{user_prompt}"
            # response = self.llm.invoke(full_prompt)
            # result = json.loads(response.content)

            self.log(f"Generated SQL query for database: {result.get('database')} and table: {result.get('table')}")
            return result

        except Exception as e:
            error_msg = f"Error generating SQL query: {e}"
            self.log(error_msg)
            return {
                "database": None,
                "table": None,
                "sql_query": None,
                "explanation": error_msg,
                "error": True
            }

    def query_database(self, user_question: str, chat_history: List = [], schema: str = None) -> Dict:
        """
        Main method to answer user questions by querying the database
        :param user_question: The user's question
        :param chat_history: Previous conversation context
        :param schema: Optional pre-fetched schema (to avoid fetching every time)
        :return: Dict with query results and explanation
        """
        self.log(f"Processing database query request: '{user_question[:50]}...'")
        self.log(f"Current db_connection_params: {bool(self.db_connection_params and self.db_connection_params.get('host'))}")

        # Use provided schema or fetch if not provided
        if schema is None:
            self.log("No schema provided, fetching from database")
            schema = self.get_schema()
        else:
            self.log("Using provided cached schema")

        if "❌" in schema or "No user databases" in schema:
            return {
                "success": False,
                "error": schema,
                "user_message": "Unable to connect to database or no databases available"
            }

        db_map = self.get_all_databases(self.db_connection_params)

        # Generate SQL from question
        sql_info = self.generate_sql_from_question(user_question, schema, chat_history)

        if sql_info.get("error") or not sql_info.get("database"):
            return {
                "success": False,
                "error": sql_info.get("explanation", "Could not generate SQL query"),
                "user_message": "I couldn't generate a valid SQL query for your question. Please try rephrasing or ask about specific data."
            }

        # Execute the query
        # query_results = self.execute_sql_query(
        #     sql_info["sql_query"],
        #     sql_info["database"],
        #     db_map
        # )

        # if not query_results.get("success"):
        #     return {
        #         "success": False,
        #         "error": query_results.get("error"),
        #         "sql_query": sql_info["sql_query"],
        #         "database": sql_info["database"],
        #         "user_message": f"Query execution failed: {query_results.get('error')}"
        #     }

        # # Format results for user
        # self.log("Successfully executed database query")
        return {
            "success": True,
            "database": sql_info["database"],
            "table": sql_info["table"],
            "sql_query": sql_info["sql_query"],
            "explanation": sql_info["explanation"],
            "assumptions": sql_info.get("assumptions", []),
            "user_message": self._format_results_for_user({}, sql_info) # change {} to query_results when enabling execution
        }

    def _format_results_for_user(self, query_results: Dict, sql_info: Dict) -> str:
        """Format query results into a user-friendly message"""
        # if not query_results.get("rows"):
        #     return f"Query executed successfully but returned no results.\n\n**Query**: `{sql_info['sql_query']}`"

        # row_count = len(query_results["rows"])
        # message = f"Found {row_count} result{'s' if row_count != 1 else ''} from database `{sql_info['database']}`:\n\n"
        message = f"**Query**: `{sql_info['sql_query']}`\n\n"
        message += f"**Explanation**: {sql_info['explanation']}\n\n"

        # Show first few results
        # display_rows = query_results["rows"][:5]
        # if display_rows:
        #     message += "**Results**:\n"
        #     for i, row in enumerate(display_rows, 1):
        #         message += f"{i}. {row}\n"

        #     if row_count > 5:
        #         message += f"\n... and {row_count - 5} more results"

        return message
