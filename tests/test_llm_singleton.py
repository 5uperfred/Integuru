import os
import unittest
from unittest.mock import patch, MagicMock
from integuru.util.LLM import LLMSingleton

class TestLLMSingleton(unittest.TestCase):

    def setUp(self):
        LLMSingleton._instance = None
        LLMSingleton._model = None
        LLMSingleton._provider = None

    @patch('integuru.util.LLM.ChatOpenAI')
    @patch.dict(os.environ, {"LLM_PROVIDER": "openai"}, clear=True)
    def test_openai_instance(self, mock_chat_openai):
        instance = LLMSingleton.get_instance()
        self.assertIs(instance, mock_chat_openai.return_value)
        mock_chat_openai.assert_called_with(model="gpt-4-turbo-preview", temperature=1)

    @patch('integuru.util.LLM.ChatGoogleGenerativeAI')
    @patch.dict(os.environ, {"LLM_PROVIDER": "gemini"}, clear=True)
    def test_gemini_instance(self, mock_chat_google):
        instance = LLMSingleton.get_instance()
        self.assertIs(instance, mock_chat_google.return_value)
        mock_chat_google.assert_called_with(model="gemini-1.5-flash", temperature=1)

    @patch('integuru.util.LLM.ChatGoogleGenerativeAI')
    @patch('integuru.util.LLM.ChatOpenAI')
    def test_singleton_pattern(self, mock_chat_openai, mock_chat_google):
        with patch.dict(os.environ, {}, clear=True):
            instance1 = LLMSingleton.get_instance()
            instance2 = LLMSingleton.get_instance()
            self.assertIs(instance1, instance2)
            self.assertIs(instance1, mock_chat_google.return_value)
            mock_chat_google.assert_called_once()
            mock_chat_openai.assert_not_called()

    @patch('integuru.util.LLM.ChatOpenAI')
    @patch.dict(os.environ, {"LLM_PROVIDER": "openai"}, clear=True)
    def test_model_change_recreates_instance(self, mock_chat_openai):
        mock_instance1 = MagicMock()
        mock_instance2 = MagicMock()
        mock_chat_openai.side_effect = [mock_instance1, mock_instance2]

        instance1 = LLMSingleton.get_instance(model="gpt-4")
        self.assertIs(instance1, mock_instance1)
        mock_chat_openai.assert_called_with(model="gpt-4", temperature=1)

        instance2 = LLMSingleton.get_instance(model="gpt-3.5-turbo")
        self.assertIs(instance2, mock_instance2)
        mock_chat_openai.assert_called_with(model="gpt-3.5-turbo", temperature=1)

        self.assertIsNot(instance1, instance2)

    def test_unsupported_provider(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "unsupported"}, clear=True):
            with self.assertRaises(ValueError):
                LLMSingleton.get_instance()

if __name__ == "__main__":
    unittest.main()
