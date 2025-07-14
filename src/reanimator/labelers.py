from abc import ABC, abstractmethod
from typing import Optional, List, Dict, TypedDict
from .models import Chunk, Topic, Document, Judgement, load_judgements
# import openai # This would be a real dependency
from openai import OpenAI
from openai import AsyncOpenAI
import re
from tqdm.asyncio import tqdm
import asyncio
from sklearn.metrics import cohen_kappa_score
import os
import pkgutil


class TopicChunkPair(TypedDict):
    topic: Topic
    chunk: Chunk


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
    async def label(self, topic: Topic, chunk: Chunk) -> Judgement:
        """
        Generates a relevance judgement for a given query and document chunk.

        Args:
            topic (Topic): The topic object.
            chunk (Chunk): The chunk object from a document.

        Returns:
            Judgement: The synthetic relevance judgement.
        """
        pass


class _BaseOpenAILabeler(BaseLabeler):
    """
    Base class for labelers using an OpenAI-compatible API. Not intended for direct use.
    """
    def __init__(self, model: str, client: AsyncOpenAI, thinking: bool = False, concurrency: int = 10, prompt_path: Optional[str] = None):
        self.model = model
        self.client = client
        if prompt_path:
            if os.path.exists(prompt_path):
                with open(prompt_path, "r") as f:
                    self.prompt_template = f.read()
            else:
                # Assume it's a package resource
                prompt_bytes = pkgutil.get_data('reanimator', prompt_path)
                if prompt_bytes:
                    self.prompt_template = prompt_bytes.decode('utf-8')
                else:
                    raise FileNotFoundError(f"Prompt file not found at path or in package: {prompt_path}")
        else:
            self.prompt_template = None
        self.temperature = 0.0
        self.thinking = thinking
        self.concurrency = concurrency

    def _construct_prompt(self, query: Topic, chunk: Chunk) -> str:
        """Constructs the prompt for the LLM."""
        if not self.prompt_template:
            raise ValueError("Prompt template not set. Please provide a `prompt_path` during initialization.")
        return self.prompt_template.replace("{query}", query.query_text).replace("{passage}", chunk.text)

    async def generate_response(self, user_input: str) -> str:
        """Generates a response from the OpenAI-compatible API."""
        if self.thinking:
            system_message = "You are a concise assistant."
        else:
            system_message = "You are a concise assistant. /no_think"
        request_params = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_input},
            ],
            "temperature": self.temperature
        }
        chat_response = await self.client.chat.completions.create(**request_params)
        return chat_response.choices[0].message.content

    async def label(self, topic: Topic, chunk: Chunk) -> Judgement:
        """
        Calls the OpenAI API to get a relevance score.
        """
        if not chunk.text:
            return Judgement(topic.query_id, chunk.doc_id,chunk_id=chunk.chunk_id, score=0, source=f"synthetic-{self.model}-skipped")

        prompt = self._construct_prompt(topic, chunk)
        response_text = await self.generate_response(prompt)
        score = parse_fewshot_response(response_text)
        return Judgement(topic.query_id, chunk.doc_id, chunk_id=chunk.chunk_id, score=score, source=f"synthetic-{self.model}")

    async def label_all(self, pairs: List[TopicChunkPair]) -> List[Judgement]:
        """
        Generates relevance judgements for a list of (topic, chunk) pairs asynchronously.

        Args:
            pairs (List[TopicChunkPair]): A list of topic-chunk pairs.

        Returns:
            List[Judgement]: A list of generated relevance judgements.
        """
        semaphore = asyncio.Semaphore(self.concurrency)

        async def sem_label(pair: TopicChunkPair) -> Judgement:
            async with semaphore:
                return await self.label(pair["topic"], pair["chunk"])

        tasks = [sem_label(pair) for pair in pairs]
        judgements = await tqdm.gather(*tasks, desc="Generating Judgements")
        return judgements


class OpenAILabeler(_BaseOpenAILabeler):
    """
    A labeler that uses an OpenAI model (like GPT-4) to generate judgements.
    """
    def __init__(self, model: str = "gpt-4.1-mini-2025-04-14", api_key: Optional[str] = None, concurrency: int = 10, thinking: bool = False, prompt_path: Optional[str] = None):
        """
        Initializes the OpenAI client.
        
        Args:
            model (str): The name of the OpenAI model to use.
            api_key (str): Your OpenAI API key. If None, it will be read from the
                           OPENAI_API_KEY environment variable.
            concurrency (int): The maximum number of concurrent requests to make.
            thinking (bool): Whether to enable 'thinking' mode for the model.
            prompt_path (str, optional): Path to a custom prompt template file. Defaults to None.
        """
        client = AsyncOpenAI(api_key=api_key)
        super().__init__(model=model, client=client, concurrency=concurrency, thinking=thinking, prompt_path=prompt_path)
        print(f"INFO: OpenAILabeler initialized with model: {self.model}")


class LocalModelLabeler(_BaseOpenAILabeler):
    """
    A labeler that uses a local model served via an OpenAI-compatible API.
    """
    def __init__(self, model: str, base_url: str, concurrency: int = 10, thinking: bool = False, prompt_path: Optional[str] = None):
        """
        Initializes the client to connect to a local model.
        
        Args:
            model (str): The name of the model to use (can be arbitrary for local models).
            base_url (str): The base URL of the local model server 
                           (e.g., "http://localhost:1234/v1").
            concurrency (int): The maximum number of concurrent requests to make.
            thinking (bool): Whether to enable 'thinking' mode for the model.
            prompt_path (str, optional): Path to a custom prompt template file. Defaults to None.
        """
        # The api_key can be a dummy value for local models.
        client = AsyncOpenAI(base_url=base_url, api_key="not-needed")
        super().__init__(model=model, client=client, concurrency=concurrency, thinking=thinking, prompt_path=prompt_path)
        print(f"INFO: LocalModelLabeler initialized with model: {self.model} at {base_url}")


def calculate_cohens_kappa(judgements_path1: str, judgements_path2: str) -> float:
    """
    Calculates Cohen's Kappa for two sets of judgements.

    This function loads two sets of relevance judgements from the given file paths,
    finds the judgements for common (query_id, doc_id) pairs, and then calculates
    Cohen's Kappa score to measure the inter-rater agreement.

    Args:
        judgements_path1 (str): Path to the first judgements file.
        judgements_path2 (str): Path to the second judgements file.

    Returns:
        float: The Cohen's Kappa score.
    """
    # Load the two sets of judgements
    judgements1 = load_judgements(judgements_path1)
    judgements2 = load_judgements(judgements_path2)

    # Create dictionaries for faster lookup, mapping (query_id, doc_id) to score
    scores1 = {(j.query_id, j.doc_id): j.score for j in judgements1}
    scores2 = {(j.query_id, j.doc_id): j.score for j in judgements2}

    # Find common keys (query_id, doc_id pairs)
    common_keys = set(scores1.keys()).intersection(set(scores2.keys()))

    if not common_keys:
        print("Warning: No common judgements found between the two files.")
        return 0.0

    # Create lists of scores for the common judgements
    rater1_scores = [scores1[key] for key in common_keys]
    rater2_scores = [scores2[key] for key in common_keys]

    # Calculate and return Cohen's Kappa
    kappa_score = cohen_kappa_score(rater1_scores, rater2_scores)
    
    print(f"Found {len(common_keys)} common judgements.")
    print(f"Cohen's Kappa: {kappa_score}")

    return kappa_score
