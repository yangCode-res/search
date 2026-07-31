import os
import unittest
from unittest.mock import patch

from pnsearch.config import Settings


class SettingsTest(unittest.TestCase):
    def test_mimo_controller_environment_is_supported(self):
        environment = {
            "CL_GISM_CONTROLLER_BASE_URL": "https://mimo.example/v1",
            "CL_GISM_CONTROLLER_API_KEY": "secret",
            "CL_GISM_CONTROLLER_MODEL": "mimo-v2.5-pro",
        }
        with patch.dict(os.environ, environment, clear=True):
            settings = Settings.from_env()
        self.assertEqual(settings.llm_base_url, "https://mimo.example/v1")
        self.assertEqual(settings.llm_api_key, "secret")
        self.assertEqual(settings.reasoner_model, "mimo-v2.5-pro")
        self.assertEqual(settings.reranker_model, "mimo-v2.5-pro")

    def test_pnsearch_environment_overrides_mimo_controller(self):
        environment = {
            "PNSEARCH_LLM_BASE_URL": "https://pnsearch.example/v1",
            "PNSEARCH_REASONER_MODEL": "reasoner",
            "CL_GISM_CONTROLLER_BASE_URL": "https://mimo.example/v1",
            "CL_GISM_CONTROLLER_MODEL": "mimo-v2.5-pro",
        }
        with patch.dict(os.environ, environment, clear=True):
            settings = Settings.from_env()
        self.assertEqual(settings.llm_base_url, "https://pnsearch.example/v1")
        self.assertEqual(settings.reasoner_model, "reasoner")
        self.assertEqual(settings.reranker_model, "mimo-v2.5-pro")
