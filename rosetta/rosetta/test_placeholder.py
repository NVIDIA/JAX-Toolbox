import pytest

def test_placeholder():
    assert 1 == 1

@pytest.mark.integration
def test_integration_placeholder():
    assert 1 == 1
