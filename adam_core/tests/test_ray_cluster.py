import pytest

from adam_core.ray_cluster import _determine_ray_memory, initialize_ray


def test_determine_ray_memory():
    # Test that if no memory is requested, we use ray default
    assert _determine_ray_memory(None) is None

    # Test that if memory is requested, but not available, we use available
    assert _determine_ray_memory(1000) == 1000

    # Test that if memory is requested, but not available, we use available
    too_much = 10000000000000000000
    have = _determine_ray_memory(too_much)
    assert have < too_much


# Test that we are calling the correct ray init function using mocks
def test_initialize_ray(mocker):
    mock_ray = mocker.patch("adam_core.ray_cluster.ray")

    # Ensure we initialize ray with the correct values
    initialize_ray(num_cpus=4, object_store_bytes=1000)
    mock_ray.init.assert_called_once_with(num_cpus=4, object_store_memory=1000)

    # Ensure that if there is a ray cluster we connect to it with address="auto"
    mock_ray.init.reset_mock()
    initialize_ray()

    mock_ray.init.assert_called_once_with(address="auto")
    assert mock_ray.init.call_count == 1
