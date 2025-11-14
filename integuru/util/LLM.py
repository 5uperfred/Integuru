import os
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

class LLMSingleton:
    _instance = None
    _model = None
    _provider = None

    @classmethod
    def get_instance(cls, model: str = None):
        provider = os.getenv("LLM_PROVIDER", "gemini")

        default_models = {
            "openai": "gpt-4-turbo-preview",
            "gemini": "gemini-1.5-flash"
        }

        if model is None:
            model = default_models.get(provider)

        if cls._instance is None or cls._provider != provider or cls._model != model:
            cls._provider = provider
            cls._model = model
            if provider == "openai":
                cls._instance = ChatOpenAI(model=model, temperature=1)
            elif provider == "gemini":
                cls._instance = ChatGoogleGenerativeAI(model=model, temperature=1)
            else:
                raise ValueError(f"Unsupported LLM provider: {provider}")

        return cls._instance
