"""
Tests for BackendConfig and WeightingPolicy.
"""

import pytest

from adam_core.orbit_determination.config import BackendConfig, WeightingPolicy


class TestWeightingPolicy:
    def test_values(self):
        assert WeightingPolicy.DELEGATE.value == "delegate"
        assert WeightingPolicy.ADAM.value == "adam"

    def test_membership(self):
        assert WeightingPolicy("delegate") is WeightingPolicy.DELEGATE
        assert WeightingPolicy("adam") is WeightingPolicy.ADAM

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            WeightingPolicy("unknown")


class TestBackendConfig:
    def test_defaults(self):
        cfg = BackendConfig()
        assert cfg.weighting_policy is WeightingPolicy.DELEGATE
        assert cfg.backend_kwargs == {}

    def test_custom_weighting_policy(self):
        cfg = BackendConfig(weighting_policy=WeightingPolicy.ADAM)
        assert cfg.weighting_policy is WeightingPolicy.ADAM

    def test_backend_kwargs_independent(self):
        """Each instance must have its own backend_kwargs dict."""
        cfg1 = BackendConfig()
        cfg2 = BackendConfig()
        cfg1.backend_kwargs["key"] = "value"
        assert "key" not in cfg2.backend_kwargs

    def test_backend_kwargs_roundtrip(self):
        cfg = BackendConfig(backend_kwargs={"propagator": object, "min_obs": 5})
        assert cfg.backend_kwargs["min_obs"] == 5
