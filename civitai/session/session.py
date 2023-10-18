from typing import Mapping, Optional

import requests

from .config import CIVITAI_ROOT
from ..utils import get_session, load_cookies, CookiesTyping


def _get_civitai_session_from_cookies(cookies: Mapping[str, str]) -> requests.Session:
    session = get_session()
    session.cookies.update(cookies)
    session.headers.update({'Referer': CIVITAI_ROOT})
    return session


def load_civitai_session(anything: Optional[CookiesTyping] = None) -> requests.Session:
    if anything:
        return _get_civitai_session_from_cookies(load_cookies(anything))
    else:
        return _get_civitai_session_from_cookies({})
