import json
from typing import Optional

import requests

from .superjs import resp_data_parse, req_data_format, undefined
from ..session import load_civitai_session, whoami, WhoAmI, CIVITAI_ROOT
from ..utils import CookiesTyping


class CivitAIClient:
    def __init__(self, session: requests.Session):
        self._session = session
        self._whoami_value = None

    @classmethod
    def load(cls, anything: Optional[CookiesTyping] = None) -> 'CivitAIClient':
        return cls(load_civitai_session(anything))

    @property
    def whoami(self) -> Optional[WhoAmI]:
        if self._whoami_value is None:
            self._whoami_value = whoami(self._session)
        return self._whoami_value

    @property
    def _authed(self) -> bool:
        return bool(self.whoami)

    def creator_info(self, creator_name):
        resp = self._session.get(
            f'{CIVITAI_ROOT}/api/trpc/user.getCreator',
            params={
                'input': json.dumps(req_data_format({"username": creator_name, "authed": self._authed})),
            },
        )
        resp.raise_for_status()

        json_ = resp.json()
        return resp_data_parse(json_['result']['data'])

    def creator_info_self(self):
        return self.creator_info(self.whoami.username)

    def buzz_count(self):
        resp = self._session.get(
            f'{CIVITAI_ROOT}/api/trpc/buzz.getUserAccount',
            params={
                'input': json.dumps(req_data_format(undefined)),
            },
        )
        resp.raise_for_status()

        json_ = resp.json()
        return resp_data_parse(json_['result']['data'])

    def iter_articles(self, username):
        next_cursor = undefined
        while True:
            resp = self._session.get(
                f'{CIVITAI_ROOT}/api/trpc/article.getInfinite',
                params={
                    "input": json.dumps(req_data_format({
                        "period": "AllTime",
                        "periodMode": "published",
                        "sort": "Newest",
                        "view": "categories",
                        "username": username,
                        "includeDrafts": False,
                        "browsingMode": "NSFW",
                        "cursor": next_cursor,
                        "authed": self._authed
                    }))
                },
            )
            resp.raise_for_status()

            json_ = resp.json()
            resp_data = resp_data_parse(json_['result']['data'])
            items = resp_data['items']
            next_cursor = resp_data['nextCursor']
            yield from items

            if next_cursor is None:
                break

    def iter_articles_self(self):
        yield from self.iter_articles(self.whoami.username)

    def iter_models(self, username):
        next_cursor = undefined
        while True:
            resp = self._session.get(
                f'{CIVITAI_ROOT}/api/trpc/model.getAll',
                params={
                    "input": json.dumps(req_data_format({
                        "period": "AllTime",
                        "periodMode": "published",
                        "sort": "Newest",
                        "view": "feed",
                        "username": username,
                        "cursor": next_cursor,
                        "authed": self._authed,
                    }))
                },
            )
            resp.raise_for_status()

            json_ = resp.json()
            resp_data = resp_data_parse(json_['result']['data'])
            items = resp_data['items']
            next_cursor = resp_data['nextCursor']
            yield from items

            if next_cursor is None:
                break

    def iter_models_self(self):
        yield from self.iter_models(self.whoami.username)
