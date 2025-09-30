import logging
from typing import Optional

import ray

logger = logging.getLogger(__name__)


def initialize_use_ray(
    num_cpus: Optional[int] = None, object_store_bytes: Optional[int] = None, **kwargs
) -> tuple[bool, int]:
    """
    Ensures we use existing local cluster, or starts new one with desired resources.

    Parameters
    ----------
    num_cpus : int, optional
        Number of CPUs to use. If None, will use all available CPUs.
    object_store_bytes : int, optional
        Amount of memory to use for the object store. If None, will use all available memory.
    **kwargs : dict, optional
        Additional keyword arguments to pass to ray.init.

    Returns
    -------
    use_ray : bool
        True if ray is initialized, False otherwise.
    num_cpus : int
        Number of CPUs used (if already initialized this will be the number of CPUs in the cluster
        regardless of the value of num_cpus passed to this function).
    """
    use_ray = False
    if num_cpus is None or num_cpus > 1:
        # Initialize ray
        if not ray.is_initialized():
            logger.info("Ray is not initialized. Initializing...")
            # For some reason, ray does not seem to automatically
            # find existing local clusters without `address="auto"`
            # but it will fail if we use auto and there is no existing cluster.
            # So we wrap it in a try/except, using an existing cluster if we can
            # Otherwise starting fresh.
            try:
                logger.info("Attempting to connect to existing ray cluster...")
                ray.init(address="auto")
            except ConnectionError:
                logger.info("Could not connect to existing ray cluster.")
                logger.info(
                    f"Attempting ray with {num_cpus} cpus and {object_store_bytes} bytes."
                )
                ray.init(
                    dashboard_host="0.0.0.0",
                    num_cpus=num_cpus,
                    object_store_memory=object_store_bytes,
                    **kwargs,
                )

        logger.info(f"Ray Resources: {ray.cluster_resources()}")

        num_cpus_effective = int(ray.cluster_resources()["CPU"])
        if num_cpus is not None and num_cpus_effective != num_cpus:
            logger.info(
                f"Requested {num_cpus} CPUs but found {num_cpus_effective} CPUs in the cluster."
            )
        use_ray = True
    else:
        num_cpus_effective = num_cpus

    return use_ray, num_cpus_effective
