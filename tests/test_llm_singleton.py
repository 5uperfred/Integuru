import os
import unittest
from unittest.mock import patch, MagicMock
from integuru.util.LLM import LLMSingleton
from langchain_google_genai import ChatGoogleGenerativeAI

class TestLLMSingleton(unittest.TestCase):

    def setUp(self):
        # Reset the singleton instance before each test
        LLMSingleton._instance = None
        LLMSingleton._model = None
        # _provider is no longer used, so no need to reset

    @patch('integuru.util.LLM.ChatGoogleGenerativeAI')
    def test_gemini_default_instance(self, mock_chat_google):
        """Test if the default instance is gemini-2.5-flash."""
        mock_chat_google.return_value = MagicMock(spec=ChatGoogleGenerativeAI)
        
        instance = LLMSingleton.get_instance()
        
        self.assertIs(instance, mock_chat_google.return_value)
        # Check that it's called with the default model and thinking config
        mock_chat_google.assert_called_with(
            model="gemini-2.5-flash",
            temperature=1,
            generation_config=LLMSingleton._thinking_config
        )

    @patch('integuru.util.LLM.ChatGoogleGenerativeAI')
    def test_singleton_pattern(self, mock_chat_google):
        """Test if get_instance() returns the same instance."""
        mock_chat_google.return_value = MagicMock(spec=ChatGoogleGenerativeAI)
        
        instance1 = LLMSingleton.get_instance()
        instance2 = LLMSingleton.get_instance()
        
        self.assertIs(instance1, instance2)
        # Ensure the model is initialized only once
        mock_chat_google.assert_called_once()

    @patch('integuru.util.LLM.ChatGoogleGenerativeAI')
    def test_model_change_recreates_instance(self, mock_chat_google):
        """Test if changing the model creates a new instance."""
        mock_instance1 = MagicMock(spec=ChatGoogleGenerativeAI)
        mock_instance2 = MagicMock(spec=ChatGoogleGenerativeAI)
        mock_chat_google.side_effect = [mock_instance1, mock_instance2]

        # Get the default instance (gemini-2.5-flash)
        instance1 = LLMSingleton.get_instance()
        self.assertIs(instance1, mock_instance1)
        mock_chat_google.assert_called_with(
            model="gemini-2.5-flash",
            temperature=1,
            generation_config=LLMSingleton._thinking_config
        )

        # Request a different model
        instance2 = LLMSingleton.get_instance(model="gemini-2.5-pro")
        self.assertIs(instance2, mock_instance2)
        mock_chat_google.assert_called_with(
            model="gemini-2.5-pro",
            temperature=1,
            generation_config=LLMSingleton._thinking_config
        )
        
        # The instances should be different
        self.assertIsNot(instance1, instance2)
        self.assertEqual(mock_chat_google.call_count, 2)

    @patch('integuru.util.LLM.ChatGoogleGenerativeAI')
    def test_code_generation_instance(self, mock_chat_google):
        """Test if the code gen instance uses gemini-2.5-pro."""
        mock_pro_instance = MagicMock(spec=ChatGoogleGenerativeAI)
        mock_chat_google.return_value = mock_pro_instance
        
        code_instance = LLMSingleton.get_code_generation_instance()
        
        self.assertIs(code_instance, mock_pro_instance)
        mock_chat_google.assert_called_with(
            model="gemini-2.5-pro",
            temperature=1,
            generation_config=LLMSingleton._thinking_config
        )

    @patch('integuru.util.LLM.ChatGoogleGenerativeAI')
    def test_code_gen_fallback(self, mock_chat_google):
        """Test if code gen instance falls back to flash if pro fails."""
        mock_flash_instance = MagicMock(spec=ChatGoogleGenerativeAI)
        
        # First call (gemini-2.5-pro) fails, second call (get_instance()) succeeds
        mock_chat_google.side_effect = [
            Exception("API Error"),  # This simulates the "try" block failing
            mock_flash_instance      # This is the "except" block calling get_instance()
        ]

        code_instance = LLMSingleton.get_code_generation_instance()
        
        # It should return the flash instance
        self.assertIs(code_instance, mock_flash_instance)
        
        # Check the call arguments
        call_list = mock_chat_google.call_args_list
        self.assertEqual(len(call_list), 2)
        
        # First call was the attempt for "pro"
        self.assertEqual(call_list[0].kwargs['model'], "gemini-2.5-pro")
        
        # Second call was the fallback to "flash"
        self.assertEqual(call_list[1].kwargs['model'], "gemini-2.5-flash")

if __name__ == "__main__":
    unittest.main()
