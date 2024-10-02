import pytest
from unittest.mock import Mock
from sfx.effect_container import EffectContainer
from sfx.effect import Effect


# Test 1: Verify that the container runs all effects successfully
def test_effect_container_success():
    # Create mock effects
    mock_effect1 = Mock(spec=Effect)
    mock_effect2 = Mock(spec=Effect)

    # Simulate successful runs
    mock_effect1.run.return_value = None  # Success case
    mock_effect2.run.return_value = None  # Success case

    # Create the container and add effects
    container = EffectContainer()
    container.append(mock_effect1)
    container.append(mock_effect2)

    # Run the container and check if all effects completed successfully
    assert container.run() == True

    # Ensure that each effect's run method was called
    mock_effect1.run.assert_called_once()
    mock_effect2.run.assert_called_once()


def test_effect_container_failure():
    # Create mock effects
    mock_effect1 = Mock(spec=Effect)
    mock_effect2 = Mock(spec=Effect)

    # Simulate one successful run and one failure
    mock_effect1.run.return_value = None  # Success
    mock_effect2.run.side_effect = Exception("Error")  # Simulate failure

    # Create the container and add effects
    container = EffectContainer()
    container.append(mock_effect1)
    container.append(mock_effect2)

    # Run the container and check if failure is detected
    assert container.run() == False

    # Ensure that both effects' run methods were called
    mock_effect1.run.assert_called_once()
    mock_effect2.run.assert_called_once()


# Test 3: Verify that only Effect instances can be appended
def test_effect_container_append_invalid():
    # Create the container
    container = EffectContainer()

    # Attempt to append a non-Effect object, expecting a ValueError
    with pytest.raises(ValueError):
        container.append("Not an effect")


# Test 4: Verify that an empty container runs successfully
def test_effect_container_empty():
    # Create an empty container
    container = EffectContainer()

    # Run the empty container and verify that it returns True
    assert container.run() == True
