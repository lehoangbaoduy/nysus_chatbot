import os
import json
from typing import Optional, List
# OpenAI import (commented out for cost savings during testing)
# from openai import OpenAI
from langchain_ollama import ChatOllama
from agents.base_classes import ScannerAgentResponse
from agents.agent import Agent


class ScannerAgent(Agent):

    # OpenAI model (commented out)
    # MODEL = "gpt-4o-mini"
    # Ollama model (using for testing)
    MODEL = "llama3.2"  # or "mistral", "qwen2.5", etc.
    MEMORY_FILE = "memory.json"

    SYSTEM_PROMPT = """You identify and summarize the 5 most detailed tickets from a list, by selecting tickets that have the most detailed, high quality description and the most clear resolution.
    Respond strictly in JSON with no explanation, using this format. You should provide the ticket number as a number derived from the description that might start with #XXXXX. If the ticket number and its content aren't clear, do not include that ticket in your response.
    Most important is that you respond with the 5 tickets that have the most detailed responses and resolution. It's not important to mention the terms of the ticket; most important is a thorough description of the ticket and what the agent did to resolve the issue.
    Only respond with tickets when you are highly confident about the resolution.

    {"questions": [
        {
            "question": "Your clearly expressed summary of the questions in 2-3 sentences. Details of the question are much more important. There should be a paragraph of text for each question you choose.",
            "answer": "The clear resolution of the question.",
            "database": {
                "relevant_database_1": "table_1",
                "relevant_database_2": "table_2"
            },
            "resolution": "True or False depending on whether the question was resolved"
        },
        ...
    ]}"""

    USER_PROMPT_PREFIX = """Respond with the most promising 5 questions from this list, selecting those which have the most detailed, high quality question description and a clear resolution.
    Respond strictly in JSON, and only JSON. You should rephrase the description to be a summary of the question itself.
    Remember to respond with a paragraph of text in the question_description field for each of the 5 items that you select.
    Only respond with questions when you are highly confident about the resolution.

    User question:

    """

    USER_PROMPT_SUFFIX = "\n\nStrictly respond in JSON and include exactly 5 questions, no more."
    name = "Scanner Agent"
    color = Agent.CYAN

    def __init__(self):
        """
        Set up this instance by initializing Ollama (or OpenAI when uncommented)
        """
        self.log("Scanner Agent is initializing")
        # OpenAI initialization (commented out for cost savings)
        # self.openai = OpenAI()
        # Ollama initialization
        self.llm = ChatOllama(model=self.MODEL, temperature=0, format="json")
        self.log("Scanner Agent is ready")

    def make_user_prompt(self, user_query, cached_questions) -> str:
        """
        Create a user prompt for OpenAI based on the cached questions from memory
        """
        user_prompt = self.USER_PROMPT_PREFIX
        user_prompt += user_query
        user_prompt += "\n\nHere are the cached recently asked questions to choose from:\n\n"
        for i, question in enumerate(cached_questions, 1):
            # Format the cached question data for the prompt
            question_text = json.dumps(question, indent=2)
            user_prompt += f"Cached Question {i}:\n{question_text}\n\n"
        user_prompt += self.USER_PROMPT_SUFFIX
        return user_prompt

    def scan(self, memory: List[dict]=[], user_query: str = "", uploaded_files: List = []) -> Optional[ScannerAgentResponse]:
        """
        Scan through cached recently asked questions to find ones that match the user query
        Use OpenAI to select the most relevant ones
        :param memory: a list of cached recently asked questions (dictionaries)
        :param user_query: the user's question
        :return: a selection of good questions, or None if there aren't any
        """
        if not memory:
            self.log("No cached recently asked questions provided in memory parameter")
            return None

        self.log(f"Scanner Agent received {len(memory)} cached recently asked questions")

        # Call LLM to select the most relevant ones based on the user query
        user_prompt = self.make_user_prompt(user_query, memory)
        self.log("Scanner Agent is calling LLM to select relevant cached recently asked questions")
        try:
            # OpenAI approach (commented out for cost savings)
            # result = self.openai.chat.completions.create(
            #     model=self.MODEL,
            #     messages=[
            #         {"role": "system", "content": self.SYSTEM_PROMPT},
            #         {"role": "user", "content": user_prompt}
            #     ],
            #     response_format=ScannerAgentResponse
            # )
            # result = result.choices[0].message.parsed

            # Ollama approach (for testing)
            full_prompt = f"{self.SYSTEM_PROMPT}\n\n{user_prompt}"
            response = self.llm.invoke(full_prompt)
            result_json = json.loads(response.content)
            result = ScannerAgentResponse(**result_json)

            self.log(f"Scanner Agent received {len(result.questions) if hasattr(result, 'questions') else 0} selected recently asked questions from LLM")
            return result
        except Exception as e:
            self.log(f"Error calling LLM: {e}")
            return None

