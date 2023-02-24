import multiprocessing as mp


def _check_parallel(num_jobs):
    """
    Helper function to determine how many workers (jobs) should be submitted.
    If num_jobs is not "auto" and instead an integer then
    that number will be used instead.

    Parameters
    ----------
    num_jobs : {"auto", int}
        Number of jobs to launch.

    Returns
    -------
    enable_parallel : bool
        True if parallelization should be used (true for cases where num_jobs > 1).
    num_workers : int
        The number of workers to use.
    """
    enable_parallel = True
    if isinstance(num_jobs, int) and num_jobs > 1:
        num_workers = num_jobs

    elif isinstance(num_jobs, str):
        if num_jobs != "auto":
            raise ValueError(f"num_jobs must be an integer or 'auto', not {num_jobs}.")
        else:
            num_workers = mp.cpu_count()

    else:
        num_workers = 1
        enable_parallel = False

    return enable_parallel, num_workers
