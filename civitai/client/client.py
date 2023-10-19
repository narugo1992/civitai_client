import json
from typing import Optional
from urllib.parse import urljoin

import requests
from hbutils.collection import nested_map
from hbutils.design import SingletonMark

from .superjs import resp_data_parse, req_data_format, undefined
from ..session import load_civitai_session, whoami, WhoAmI, CIVITAI_ROOT
from ..utils import CookiesTyping


class SessionError(Exception):
    pass


m_page = SingletonMark('mark_page')
m_cursor = SingletonMark('mark_cursor')


def _replace_page(data, page):
    return nested_map(lambda x: page if x is m_page else x, data)


def _replace_cursor(data, cursor):
    return nested_map(lambda x: cursor if x is m_cursor else x, data)


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

    def _get(self, url, data=undefined):
        resp = self._session.get(
            urljoin(CIVITAI_ROOT, url),
            params={
                'input': json.dumps(req_data_format(data)),
            },
        )
        resp.raise_for_status()
        json_ = resp.json()
        return resp_data_parse(json_['result']['data'])

    def _post(self, url, data=undefined):
        resp = self._session.post(
            urljoin(CIVITAI_ROOT, url),
            json=req_data_format(data),
        )
        resp.raise_for_status()
        json_ = resp.json()
        return resp_data_parse(json_['result']['data'])

    @classmethod
    def _iter_via_cursor_fn(cls, fn):
        cursor = undefined
        while True:
            items, cursor = fn(cursor)
            yield from items

            if cursor is None:
                break

    def _iter_via_cursor(self, url, data):
        def _fn(cursor):
            resp_data = self._get(url, _replace_cursor(data, cursor))
            return resp_data['items'], resp_data['nextCursor']

        yield from self._iter_via_cursor_fn(_fn)

    @classmethod
    def _iter_via_page_fn(cls, fn):
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

    def _iter_via_page(self, url, data):
        def _fn(page):
            resp_data = self._get(url, _replace_page(data, page))
            return resp_data['items']

        yield from self._iter_via_page_fn(_fn)

    def get_buzz_count(self):
        return self._get('/api/trpc/buzz.getUserAccount')

    def iter_buzz_transactions(self, limit: int = 200):
        return self._get(
            '/api/trpc/buzz.getUserTransactions',
            {
                "limit": limit,
                "authed": self._authed,
            }
        )['transactions']

    def iter_articles(self, username):
        yield from self._iter_via_cursor(
            '/api/trpc/article.getInfinite',
            {
                "period": "AllTime",
                "periodMode": "published",
                "sort": "Newest",
                "view": "categories",
                "username": username,
                "includeDrafts": False,
                "browsingMode": "NSFW",
                "cursor": m_cursor,
                "authed": self._authed
            }
        )

    def iter_articles_self(self):
        yield from self.iter_articles(self._username)

    def iter_models(self, username):
        yield from self._iter_via_cursor(
            '/api/trpc/model.getAll',
            {
                "period": "AllTime",
                "periodMode": "published",
                "sort": "Newest",
                "view": "feed",
                "username": username,
                "cursor": m_cursor,
                "authed": self._authed,
            }
        )

    def iter_models_self(self):
        yield from self.iter_models(self._username)

    def iter_draft_models(self, limit: int = 10):
        yield from self._iter_via_page(
            '/api/trpc/model.getMyDraftModels',
            {
                "page": m_page,
                "limit": limit,
                "authed": self._authed,
            }
        )

    def iter_training_models(self, limit: int = 10):
        yield from self._iter_via_page(
            '/api/trpc/model.getMyTrainingModels',
            {
                "page": m_page,
                "limit": limit,
                "authed": self._authed
            }
        )

    def iter_posts(self, username):
        yield from self._iter_via_cursor(
            '/api/trpc/post.getInfinite',
            {
                "period": "AllTime",
                "periodMode": "published",
                "sort": "Newest",
                "view": "categories",
                "username": username,
                "draftOnly": False,
                "browsingMode": "NSFW",
                "include": [],
                "cursor": m_cursor,
                "authed": self._authed
            }
        )

    def iter_posts_self(self):
        yield from self.iter_posts(self._username)

    def iter_images(self, username):
        yield from self._iter_via_cursor(
            '/api/trpc/image.getInfinite',
            {
                "period": "AllTime",
                "sort": "Newest",
                "view": "categories",
                "username": username,
                "withMeta": False,
                "browsingMode": "NSFW",
                "cursor": m_cursor,
                "authed": self._authed,
            }
        )

    def iter_images_self(self):
        yield from self.iter_images(self._username)

    def iter_collections(self, userid):
        yield from self._iter_via_cursor(
            '/api/trpc/collection.getInfinite',
            {
                "sort": "Newest",
                "userId": userid,
                "cursor": m_cursor,
                "authed": self._authed,
            }
        )

    def iter_collections_self(self):
        yield from self.iter_collections(self._userid)

    def get_followers(self, username):
        return self._get(
            '/api/trpc/user.getLists',
            {
                "username": username,
                "authed": self._authed,
            }
        )

    def get_followers_self(self):
        return self.get_followers(self._username)

    def iter_notifications(self):
        yield from self._iter_via_cursor(
            '/api/trpc/notification.getAllByUser',
            {
                "cursor": m_cursor,
                "authed": self._authed
            }
        )

    def mark_notification_to_read(self):
        return self._post(
            '/api/trpc/notification.markRead',
            {
                "id": undefined,
                "all": True,
                "userId": self._userid,
                "authed": self._authed,
            }
        )

    def get_creator(self, username):
        return self._get(
            '/api/trpc/user.getCreator',
            {
                "username": username,
                "authed": self._authed
            }
        )

    def get_myself(self):
        return self.get_creator(self._username)

    def get_article(self, article_id):
        pass
