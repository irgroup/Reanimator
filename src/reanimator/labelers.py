from abc import ABC, abstractmethod
from typing import Optional
from .models import Chunk, Topic, Document, Judgement
# import openai # This would be a real dependency
from openai import OpenAI
from openai import AsyncOpenAI
import re


def parse_fewshot_response(response: str):
    response = response.strip().lower()
    valid_res = 1
    answer = ""
    patterns = [
        r'"o"\s*[:-=]?\s*(0|1|2|3)',
        r"\'o\'\s*[:-=]?\s*(0|1|2|3)",
        r"o\s*[:-=]?\s*(0|1|2|3)",
        r'"overall_score"\s*[:-=]?\s*(0|1|2|3)',
        r'"overall"\s*[:-=]?\s*(0|1|2|3)',
        r'"overall score"\s*[:-=]?\s*(0|1|2|3)',
        r'"final score"\s*[:-=]?\s*(0|1|2|3)',
        r"final score\s*[:-=]?\s*(0|1|2|3)",
        r"final score is (0|1|2|3)",
        r'"final_score"\s*[:-=]?\s*(0|1|2|3)',
        r'"score"\s*[:-=]?\s*(0|1|2|3)',
        r'"o_score"\s*[:-=]?\s*(0|1|2|3)',
        r"output score is (0|1|2|3)",
        r"score is (0|1|2|3)",
        r"[a-zA-Z]+\s+is\s+(0|1|2|3)\s",
        r"relevance category\s*[:-=]?\s*(0|1|2|3)",
        r"relevance category\s*[:-=]?\s*(0|1|2|3)",
        r"relevance category is (0|1|2|3)",
        r"it falls into the category (0|1|2|3)",
        r"category\s*(0|1|2|3)",
        r"relevance category (0|1|2|3)",
        r"relevance category for this passage would be (0|1|2|3)",
        r"the relevance category would be (0|1|2|3)",
        r"\n*(0|1|2|3)",
    ]
    for pattern in patterns:
        matched = None
        for m in re.finditer(
            pattern, response, re.IGNORECASE | re.MULTILINE | re.DOTALL
        ):
            matched = m

        if matched:
            answer = matched.group(1).capitalize()
            break

    return int(answer)


class BaseLabeler(ABC):
    """Abstract base class for all labelers."""
    @abstractmethod
    def label(self, query: Topic, document: Document) -> Judgement:
        """
        Generates a relevance judgement for a given query and document.

        Args:
            query (Topic): The query object.
            document (Document): The document object, with text and/or tables.

        Returns:
            Judgement: The synthetic relevance judgement.
        """
        pass

class OpenAILabeler(BaseLabeler):
    """
    A labeler that uses an OpenAI model (like GPT-4) to generate judgements.
    """
    def __init__(self, model: str = "gpt-4.1-mini-2025-04-14", api_key: Optional[str] = None):
        """
        Initializes the OpenAI client.
        
        Args:
            model (str): The name of the OpenAI model to use.
            api_key (str): Your OpenAI API key. If None, it should be set as an
                           environment variable (OPENAI_API_KEY).
        """
        self.model = model
        self.client = AsyncOpenAI()
        # if api_key:
        #     openai.api_key = api_key
        #
        # if not openai.api_key:
        #     raise ValueError("OpenAI API key not provided or set in environment.")
        
        #Prompt taken from https://github.com/castorini/umbrela/blob/main/src/umbrela/prompts/qrel_zeroshot_bing.txt
        self.prompt_template = open("/workspace/data/prompts/default_prompt.txt", "r").read()

        print(f"INFO: OpenAILabeler initialized with model: {self.model}")

    def _construct_prompt(self, query: Topic, chunk: Chunk) -> str:
        """Constructs the prompt for the LLM."""
        # This is a placeholder. The actual prompt engineering is a critical step
        # and would be based on the logic in the auto_judging notebooks.
        prompt = self.prompt_template.replace("{query}", query.query_text).replace("{passage}", chunk.text)
        self.temperature = 0.0
        return prompt

    async def generate_response(self, user_input: str):
        
        request_params = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a concise assistant"},
                {"role": "user", "content": user_input},
            ] 
        }

        request_params["temperature"] = self.temperature

        chat_response = await self.client.chat.completions.create(**request_params)
        return chat_response.choices[0].message.content

    async def label(self, topic: Topic, chunk: Chunk) -> Judgement:
        """
        Calls the OpenAI API to get a relevance score.
        """
        if not chunk.text:
            return Judgement(topic.query_id, chunk.doc_id, score=0, source=f"synthetic-{self.model}-skipped")

        prompt = self._construct_prompt(topic, chunk)
        response = await self.generate_response(prompt)
        response = parse_fewshot_response(response)
        return Judgement(topic.query_id, chunk.doc_id, score=response, source=f"synthetic-{self.model}")