from adam_core.ray_cluster import initialize_use_ray


# Test that we are calling the correct ray init function using mocks
def test_initialize_ray(mocker):
    mock_ray = mocker.patch("adam_core.ray_cluster.ray")

    # Ensure we initialize ray with the correct values
    mock_ray.is_initialized.return_value = False
    # Set mock to throw an error if init is called with address="auto"
    # but not if called with other args
    mock_ray.init.side_effect = [
        ConnectionError,
        None,
    ]

    initialize_use_ray(num_cpus=4, object_store_bytes=1000)
    mock_ray.init.assert_called_with(address="auto")
    mock_ray.init.assert_called_with(num_cpus=4, object_store_memory=1000)
    mock_ray.init.call_count == 2

    mock_ray.reset_mock(return_value=True, side_effect=True)
    mock_ray.is_initialized.return_value = False
    mock_ray.init.return_value = True
    initialize_use_ray()

    mock_ray.init.assert_called_once_with(address="auto")
    assert mock_ray.init.call_count == 1

    mock_ray.reset_mock(return_value=True, side_effect=True)
    mock_ray.is_initialized.return_value = True
    initialize_use_ray()
    assert mock_ray.init.call_count == 0
