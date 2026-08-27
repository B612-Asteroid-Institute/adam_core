from typing import NoReturn


def _raise_compatible_http_error(error: RuntimeError) -> NoReturn:
    """Translate Rust transport failures to requests-compatible exceptions."""
    import requests

    message = str(error)
    lowered = message.lower()
    if "timed out" in lowered or "timeout" in lowered:
        raise requests.Timeout(message) from error
    if "status code" in lowered or "http status" in lowered:
        raise requests.HTTPError(message) from error
    raise requests.ConnectionError(message) from error
