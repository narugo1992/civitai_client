import copy
import datetime
import json
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Mapping

import dateparser
from pyquery import PyQuery as pq


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
        return get_whoami_by_raw_user_info(user_json)
    else:
        return None


def get_whoami_by_raw_user_info(raw_user_info: Optional[Mapping[str, Any]]) -> Optional[WhoAmI]:
    if raw_user_info:
        return WhoAmI(
            id=raw_user_info['id'],
            name=raw_user_info.get('name'),
            username=raw_user_info['username'],
            email=raw_user_info['email'],
            icon_image_url=raw_user_info.get('image'),
            created_at=dateparser.parse(raw_user_info['createdAt']),
            raw=copy.deepcopy(dict(raw_user_info or {})),
        )
    else:
        return None
