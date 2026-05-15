"""Shared pytest fixtures for the test suite."""
import pytest
import torch


@pytest.fixture
def device():
    """Return the default device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def dummy_input(device):
    """Return a standard dummy input tensor for image models."""
    return torch.randn(2, 3, 64, 64, device=device)


@pytest.fixture
def dummy_classification_input(device):
    """Return a dummy input for classification models."""
    return torch.randn(4, 3, 32, 32, device=device)


@pytest.fixture
def dummy_linear_input(device):
    """Return a dummy input for linear/fully-connected layers."""
    return torch.randn(8, 64, device=device)
