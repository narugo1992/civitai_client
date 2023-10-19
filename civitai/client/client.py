import json
from functools import partial
from typing import Optional

import requests

from .superjs import resp_data_parse, req_data_format, undefined
from ..session import load_civitai_session, whoami, WhoAmI, CIVITAI_ROOT
from ..utils import CookiesTyping


class SessionError(Exception):
    pass


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

    @property
    def _username(self) -> str:
        if self.whoami:
            return self.whoami.username
        else:
            raise SessionError('You need to login first.')

    @property
    def _userid(self) -> int:
        if self.whoami:
            return self.whoami.id
        else:
            raise SessionError('You need to login first.')

    def creator_info(self, creator_name):
        resp = self._session.get(
            f'{CIVITAI_ROOT}/api/trpc/user.getCreator',
            params={
                'input': json.dumps(req_data_format({
                    "username": creator_name,
                    "authed": self._authed
                })),
            },
        )
        resp.raise_for_status()

        json_ = resp.json()
        return resp_data_parse(json_['result']['data'])

    def creator_info_self(self):
        return self.creator_info(self._username)

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

    @classmethod
    def _iter_via_cursor(cls, fn):
        cursor = undefined
        while True:
            items, cursor = fn(cursor)
            yield from items

            if cursor is None:
                break

    @classmethod
    def _iter_via_page(cls, fn):
        page = 1
        while True:
            items = fn(page)
            is_empty = True
            for item in items:
                yield item
                is_empty = False

            if is_empty:
                break
            page += 1

    def _iter_articles(self, username, next_cursor):
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
        return items, next_cursor

    def iter_articles(self, username):
        yield from self._iter_via_cursor(partial(self._iter_articles, username))

    def iter_articles_self(self):
        yield from self.iter_articles(self._username)

    def _iter_models(self, username, next_cursor):
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
        return items, next_cursor

    def iter_models(self, username):
        yield from self._iter_via_cursor(partial(self._iter_models, username))

    def iter_models_self(self):
        yield from self.iter_models(self._username)

    def _iter_draft_models(self, limit: int, page: int):
        resp = self._session.get(
            'https://civitai.com/api/trpc/model.getMyDraftModels',
            params={
                'input': json.dumps(req_data_format({
                    "page": page, "limit": limit,
                    "authed": self._authed,
                }))
            }
        )
        resp.raise_for_status()
        json_ = resp.json()
        resp_data = resp_data_parse(json_['result']['data'])
        items = resp_data['items']
        return items

    def iter_draft_models(self, limit: int = 10):
        yield from self._iter_via_page(partial(self._iter_draft_models, limit))

    def _iter_training_models(self, limit: int, page: int):
        resp = self._session.get(
            'https://civitai.com/api/trpc/model.getMyTrainingModels',
            params={
                'input': json.dumps(req_data_format({
                    "page": page, "limit": limit,
                    "authed": self._authed
                })),
            }
        )
        resp.raise_for_status()
        json_ = resp.json()
        resp_data = resp_data_parse(json_['result']['data'])
        items = resp_data['items']
        return items

    def iter_training_models(self, limit: int = 10):
        yield from self._iter_via_page(partial(self._iter_training_models, limit))

    def _iter_posts(self, username, cursor):
        resp = self._session.get(
            'https://civitai.com/api/trpc/post.getInfinite',
            params={
                'input': json.dumps(req_data_format({
                    "period": "AllTime",
                    "periodMode": "published",
                    "sort": "Newest",
                    "view": "categories",
                    "username": username,
                    "draftOnly": False,
                    "browsingMode": "NSFW",
                    "include": [],
                    "cursor": cursor,
                    "authed": self._authed
                }))
            }
        )
        resp.raise_for_status()
        json_ = resp.json()
        resp_data = resp_data_parse(json_['result']['data'])
        items = resp_data['items']
        next_cursor = resp_data['nextCursor']
        return items, next_cursor

    def iter_posts(self, username):
        yield from self._iter_via_cursor(partial(self._iter_posts, username))

    def iter_posts_self(self):
        yield from self.iter_posts(self._username)

    def _iter_images(self, username, cursor):
        resp = self._session.get(
            'https://civitai.com/api/trpc/image.getInfinite',
            params={
                'input': json.dumps(req_data_format({
                    "period": "AllTime",
                    "sort": "Newest",
                    "view": "categories",
                    "username": username,
                    "withMeta": False,
                    "browsingMode": "NSFW",
                    "cursor": cursor,
                    "authed": self._authed,
                }))
            }
        )
        resp.raise_for_status()
        json_ = resp.json()
        resp_data = resp_data_parse(json_['result']['data'])
        items = resp_data['items']
        next_cursor = resp_data['nextCursor']
        return items, next_cursor

    def iter_images(self, username):
        yield from self._iter_via_cursor(partial(self._iter_images, username))

    def iter_images_self(self):
        yield from self.iter_images(self._username)

    def _iter_collections(self, userid, cursor):
        resp = self._session.get(
            'https://civitai.com/api/trpc/collection.getInfinite',
            params={
                'input': json.dumps(req_data_format({
                    "sort": "Newest",
                    "userId": userid,
                    "cursor": cursor,
                    "authed": self._authed,
                }))
            },
        )
        resp.raise_for_status()
        json_ = resp.json()
        resp_data = resp_data_parse(json_['result']['data'])
        items = resp_data['items']
        next_cursor = resp_data['nextCursor']
        return items, next_cursor

    def iter_collections(self, userid):
        yield from self._iter_via_cursor(partial(self._iter_collections, userid))

    def iter_collections_self(self):
        yield from self.iter_collections(self._userid)

    def get_followers(self, username):
        resp = self._session.get(
            'https://civitai.com/api/trpc/user.getLists',
            params={
                'input': json.dumps(req_data_format({
                    "username": username,
                    "authed": self._authed,
                }))
            }
        )
        resp.raise_for_status()
        json_ = resp.json()
        return resp_data_parse(json_['result']['data'])

    def get_followers_self(self):
        return self.get_followers(self._username)

    def get_notifications(self, limit: int = 10):
        resp = self._session.get(
            'https://civitai.com/api/trpc/notification.getAllByUser',
            params={
                'input': json.dumps(req_data_format({
                    "limit": limit,
                    "authed": self._authed
                }))
            }
        )
        resp.raise_for_status()
        json_ = resp.json()
        return resp_data_parse(json_['result']['data'])

    def mark_notification_to_read(self):
        resp = self._session.post(
            'https://civitai.com/api/trpc/notification.markRead',
            json=req_data_format({
                "id": undefined,
                "all": True,
                "userId": self._userid,
                "authed": self._authed,
            })
        )
        resp.raise_for_status()
        json_ = resp.json()
        return resp_data_parse(json_['result']['data'])
