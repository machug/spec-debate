"""Tests for Azure AI Foundry endpoint derivation logic."""

from __future__ import annotations

import pytest


class TestFoundryEndpointDerivation:
    """Test the endpoint URL derivation logic from call_foundry_model.

    We extract and test the logic without calling the actual Azure SDK.
    """

    @staticmethod
    def derive_endpoint(api_base: str) -> str:
        """Reproduce the endpoint derivation logic from call_foundry_model."""
        endpoint = api_base.rstrip("/")
        if not endpoint.endswith("/models"):
            parts = endpoint.split(".services.ai.azure.com")
            if len(parts) == 2:
                endpoint = parts[0] + ".services.ai.azure.com/models"
            else:
                endpoint = endpoint + "/models"
        return endpoint

    @staticmethod
    def derive_deployment_name(model: str) -> str:
        """Reproduce the deployment name derivation from call_foundry_model."""
        return model.split("/", 1)[1] if "/" in model else model

    def test_project_path_stripped(self):
        url = "https://my-resource.services.ai.azure.com/api/projects/my-project"
        assert self.derive_endpoint(url) == "https://my-resource.services.ai.azure.com/models"

    def test_already_models_endpoint(self):
        url = "https://my-resource.services.ai.azure.com/models"
        assert self.derive_endpoint(url) == "https://my-resource.services.ai.azure.com/models"

    def test_bare_resource_url(self):
        url = "https://my-resource.services.ai.azure.com"
        assert self.derive_endpoint(url) == "https://my-resource.services.ai.azure.com/models"

    def test_trailing_slash_stripped(self):
        url = "https://my-resource.services.ai.azure.com/api/projects/foo/"
        assert self.derive_endpoint(url) == "https://my-resource.services.ai.azure.com/models"

    def test_non_azure_url_appends_models(self):
        url = "https://custom-proxy.example.com/v1"
        assert self.derive_endpoint(url) == "https://custom-proxy.example.com/v1/models"

    def test_deployment_name_from_foundry_prefix(self):
        assert self.derive_deployment_name("foundry/gpt-5-mini") == "gpt-5-mini"

    def test_deployment_name_from_bare_model(self):
        assert self.derive_deployment_name("gpt-5-mini") == "gpt-5-mini"

    def test_deployment_name_preserves_dots(self):
        assert self.derive_deployment_name("foundry/gpt-4.1-mini") == "gpt-4.1-mini"
