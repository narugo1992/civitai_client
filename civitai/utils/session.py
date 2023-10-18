import threading
from functools import lru_cache
from typing import Callable, Optional, Dict

import requests
from requests.adapters import HTTPAdapter, Retry

DEFAULT_TIMEOUT = 10  # seconds

BACKEND_FACTORY_T = Callable[[], requests.Session]
_GLOBAL_BACKEND_FACTORY: BACKEND_FACTORY_T = requests.session


def configure_http_backend(backend_factory: BACKEND_FACTORY_T = requests.Session) -> None:
    """
    Configure the HTTP backend by providing a `backend_factory`. Any HTTP calls made by `huggingface_hub` will use a
    Session object instantiated by this factory. This can be useful if you are running your scripts in a specific
    environment requiring custom configuration (e.g. custom proxy or certifications).

    Use [`get_session`] to get a configured Session. Since `requests.Session` is not guaranteed to be thread-safe,
    `huggingface_hub` creates 1 Session instance per thread. They are all instantiated using the same `backend_factory`
    set in [`configure_http_backend`]. A LRU cache is used to cache the created sessions (and connections) between
    calls. Max size is 128 to avoid memory leaks if thousands of threads are spawned.

    See [this issue](https://github.com/psf/requests/issues/2766) to know more about thread-safety in `requests`.

    Example:
    ```py
    import requests
    from huggingface_hub import configure_http_backend, get_session

    # Create a factory function that returns a Session with configured proxies
    def backend_factory() -> requests.Session:
        session = requests.Session()
        session.proxies = {"http": "http://10.10.1.10:3128", "https": "https://10.10.1.11:1080"}
        return session

    # Set it as the default session factory
    configure_http_backend(backend_factory=backend_factory)

    # In practice, this is mostly done internally in `huggingface_hub`
    session = get_session()
    ```
    """
    global _GLOBAL_BACKEND_FACTORY
    _GLOBAL_BACKEND_FACTORY = backend_factory
    _get_session_from_cache.cache_clear()


def get_session() -> requests.Session:
    """
    Get a `requests.Session` object, using the session factory from the user.

    Use [`get_session`] to get a configured Session. Since `requests.Session` is not guaranteed to be thread-safe,
    `huggingface_hub` creates 1 Session instance per thread. They are all instantiated using the same `backend_factory`
    set in [`configure_http_backend`]. A LRU cache is used to cache the created sessions (and connections) between
    calls. Max size is 128 to avoid memory leaks if thousands of threads are spawned.

    See [this issue](https://github.com/psf/requests/issues/2766) to know more about thread-safety in `requests`.

    Example:
    ```py
    import requests
    from huggingface_hub import configure_http_backend, get_session

    # Create a factory function that returns a Session with configured proxies
    def backend_factory() -> requests.Session:
        session = requests.Session()
        session.proxies = {"http": "http://10.10.1.10:3128", "https": "https://10.10.1.11:1080"}
        return session

    # Set it as the default session factory
    configure_http_backend(backend_factory=backend_factory)

    # In practice, this is mostly done internally in `huggingface_hub`
    session = get_session()
    ```
    """
    return _get_session_from_cache(thread_ident=threading.get_ident())


@lru_cache(maxsize=128)  # default value for Python>=3.8. Let's keep the same for Python3.7
def _get_session_from_cache(thread_ident: int) -> requests.Session:
    """
    Create a new session per thread using global factory. Using LRU cache (maxsize 128) to avoid memory leaks when
    using thousands of threads. Cache is cleared when `configure_http_backend` is called.
    """
    _ = thread_ident
    return _GLOBAL_BACKEND_FACTORY()


class TimeoutHTTPAdapter(HTTPAdapter):
    """
    Custom HTTP adapter that sets a default timeout for requests.

    Inherits from `HTTPAdapter`.

    Usage:
    - Create an instance of `TimeoutHTTPAdapter` and pass it to a `requests.Session` object's `mount` method.

    Example:
    ```python
    session = requests.Session()
    adapter = TimeoutHTTPAdapter(timeout=10)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    ```

    :param timeout: The default timeout value in seconds. (default: 10)
    :type timeout: int
    """

    def __init__(self, *args, **kwargs):
        self.timeout = DEFAULT_TIMEOUT
        if "timeout" in kwargs:
            self.timeout = kwargs["timeout"]
            del kwargs["timeout"]
        super().__init__(*args, **kwargs)

    def send(self, request, **kwargs):
        """
        Sends a request with the provided timeout value.

        :param request: The request to send.
        :type request: PreparedRequest
        :param kwargs: Additional keyword arguments.
        :type kwargs: dict
        :returns: The response from the request.
        :rtype: Response
        """
        timeout = kwargs.get("timeout")
        if timeout is None:
            kwargs["timeout"] = self.timeout
        return super().send(request, **kwargs)


def _get_requests_session(max_retries: int = 5, timeout: int = DEFAULT_TIMEOUT,
                          headers: Optional[Dict[str, str]] = None, session: Optional[requests.Session] = None) \
        -> requests.Session:
    """
    Returns a requests Session object configured with retry and timeout settings.

    :param max_retries: The maximum number of retries. (default: 5)
    :type max_retries: int
    :param timeout: The default timeout value in seconds. (default: 10)
    :type timeout: int
    :param headers: Additional headers to be added to the session. (default: None)
    :type headers: Optional[Dict[str, str]]
    :param session: An existing requests Session object to use. If not provided, a new Session object is created. (default: None)
    :type session: Optional[requests.Session]
    :returns: The requests Session object.
    :rtype: requests.Session
    """
    session = session or requests.session()
    retries = Retry(
        total=max_retries, backoff_factor=1,
        status_forcelist=[413, 429, 500, 501, 502, 503, 504, 505, 506, 507, 509, 510, 511],
        allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "TRACE"],
    )
    adapter = TimeoutHTTPAdapter(max_retries=retries, timeout=timeout, pool_connections=32, pool_maxsize=32)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
        **dict(headers or {}),
    })

    return session


configure_http_backend(_get_requests_session)
