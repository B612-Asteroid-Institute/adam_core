import logging
from typing import Optional

import psutil
import ray

logger = logging.getLogger(__name__)


def _determine_ray_memory(requested_memory_bytes: Optional[int]) -> Optional[int]:
    # If memory bytes is left at 0, use ray default
    memory_bytes = None
    if requested_memory_bytes is None:
        logger.debug("No memory bytes requested, using ray default.")
        return None

    if requested_memory_bytes > 0:
        memory_state = psutil.virtual_memory()
        if requested_memory_bytes > memory_state.available:
            logger.warning(
                f"Requested {memory_bytes} bytes for ray, but only {memory_state.available} available."
                f" Using {memory_state.available} bytes instead."
            )
            memory_bytes = memory_state.available
    return memory_bytes


def initialize_ray(
    num_cpus: Optional[int] = None, object_store_bytes: Optional[int] = None
) -> None:
    # Initialize ray
    if not ray.is_initialized():
        logger.debug(f"Ray is not initialized...")
        # For some reason, ray does not seem to automatically
        # find existing local clusters without `address="auto"`
        # but it will fail if we use auto and there is no existing cluster.
        # So we wrap it in a try/except, using an existing cluster if we can
        # Otherwise starting fresh.
        try:
            logger.debug("Trying to connect to existing cluster...")
            ray.init(address="auto")
        except ConnectionError:
            logger.debug("No existing cluster found, starting new cluster...")
            memory_bytes = _determine_ray_memory(object_store_bytes)
            ray.init(num_cpus=num_cpus, object_store_memory=memory_bytes)
