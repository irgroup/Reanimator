import openai
from langsmith import traceable
from langsmith.wrappers import wrap_openai
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

class RagBot:
    def __init__(
        self,
        retriever,
        provider: str = "openai",  # "openai", "anthropic", or "ollama"
        model: str = "gpt-4o-mini",
        num_answers: int = 1,
        temperature: float = 0,
        openai_api_key: str = None,
        anthropic_api_key: str = None,
        ollama_base_url: str = None,
    ):
        self._retriever = retriever
        self._provider = provider
        self._model = model
        self._num_answers = num_answers
        self._temperature = temperature

        # Initialize client with explicit API keys
        if provider == "openai":
            if not openai_api_key:
                raise ValueError("OpenAI API key is required")
            openai.api_key = openai_api_key
            self._client = wrap_openai(openai.Client())
            
        elif provider == "anthropic":
            if not anthropic_api_key:
                raise ValueError("Anthropic API key is required")
            self._client = ChatAnthropic(
                model_name=model,
                temperature=temperature,
                anthropic_api_key=anthropic_api_key
            )
            
        elif provider == "ollama":
            self._client = ChatOllama(
                model=model,
                base_url=ollama_base_url,
                temperature=temperature,
            )
            
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    @traceable()
    def retrieve_docs(self, question):
        return self._retriever.invoke(question)

    @traceable()
    def invoke_llm(self, question, docs):
        system_content = (
            "You are a helpful AI assistant with expertise in COVID-19. "
            "Use the following texts and/or tables to produce a concise answer to the user question.\n\n"
            f"## Tables/Texts\n\n{docs}"
        )

        if self._provider == "openai":
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": question},
                ],
                temperature=self._temperature,
                n=self._num_answers,
            )
            #answer = response.choices[0].message.content
            answer = [choice.message.content for choice in response.choices]
            
        elif self._provider == "anthropic":
            answer = []
            for _ in range(self._num_answers):
                response = self._client.invoke(
                    [HumanMessage(content=question)],
                    system=system_content
                )
                answer.append(response.content)
            
        elif self._provider == "ollama":
            response = self._client.invoke([
                SystemMessage(content=system_content),
                HumanMessage(content=question)
            ])
            answer = response.content

        return {
            "answer": answer,
            "contexts": [str(doc) for doc in docs],
        }

    @traceable()
    def get_answer(self, question: str):
        docs = self.retrieve_docs(question)
        return self.invoke_llm(question, docs)
