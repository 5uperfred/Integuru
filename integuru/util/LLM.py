import os
# No longer importing ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

class LLMSingleton:
    _instance = None
    _model = None
    # _provider is no longer needed

    # Define the "thinking" configuration based on Google's documentation
    # This will be applied to all model instances
    _thinking_config = {
        "thinking_config": {
            "include_thoughts": True
        }
    }

    @classmethod
    def get_instance(cls, model: str = None):
        # Provider check is removed, we are only using Gemini
        
        # Set default model to gemini-2.5-flash if not specified
        if model is None:
            model = "gemini-2.5-flash"

        # Re-create instance if it's None or the model has changed
        if cls._instance is None or cls._model != model:
            cls._model = model
            
            try:
                cls._instance = ChatGoogleGenerativeAI(
                    model=model,
                    temperature=0.3,
                    # Apply the thinking configuration
                    generation_config=cls._thinking_config 
                )
            except Exception as e:
                raise ValueError(f"Failed to initialize Gemini model: {model}. Error: {e}")

        return cls._instance

    @classmethod
    def get_code_generation_instance(cls):
        # This method now returns a dedicated Gemini 2.5 Pro instance 
        # for high-quality code generation.
        
        code_model = "gemini-2.5-pro"
        
        try:
            # Create a new, separate instance for code gen
            # This does not affect the main singleton instance
            llm_for_code_gen = ChatGoogleGenerativeAI(
                model=code_model,
                temperature=0.3, # Keeping temp=1 from your original file
                generation_config=cls._thinking_config
            )
            return llm_for_code_gen
        
        except Exception as e:
            # If Pro fails (e.g., API access), fall back to the default instance
            print(f"Warning: {code_model} not available ({e}).")
            print("Falling back to the default singleton instance (gemini-2.5-flash).")
            return cls.get_instance()
