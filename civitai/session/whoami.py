import copy
import datetime
import json
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

import dateparser
import requests
from pyquery import PyQuery as pq

from .config import CIVITAI_ROOT


@dataclass
class WhoAmI:
    id: int
    name: str
    username: str
    email: str
    created_at: datetime.datetime
    icon_image_url: Optional[str] = field(repr=False, default=None)
    raw: Dict[str, Any] = field(repr=False, default_factory=dict)


def _get_whoami_by_page_source(page_source: str) -> Optional[WhoAmI]:
    metadata_text = pq(page_source)('script#__NEXT_DATA__').text()
    metadata_json = json.loads(metadata_text)
    session_json = metadata_json["props"]["pageProps"]["session"]
    if session_json:
        user_json = session_json['user']
        return WhoAmI(
            id=user_json['id'],
            name=user_json['name'],
            username=user_json['username'],
            email=user_json['email'],
            icon_image_url=user_json['image'],
            created_at=dateparser.parse(user_json['createdAt']),
            raw=copy.deepcopy(user_json),
        )
    else:
        return None


def whoami(session: requests.Session) -> Optional[WhoAmI]:
    resp = session.get(f'{CIVITAI_ROOT}')
    resp.raise_for_status()
    return _get_whoami_by_page_source(resp.text)
