import json
from typing import Optional
from urllib.parse import urljoin

import requests
from hbutils.collection import nested_map
from hbutils.design import SingletonMark

from .exceptions import SessionError, APIError
from .superjs import resp_data_parse, req_data_format, undefined
from ..session import load_civitai_session, whoami, WhoAmI, CIVITAI_ROOT
from ..utils import CookiesTyping

try:
    from typing import Literal
except (ImportError, ModuleNotFoundError):
    from typing_extensions import Literal

m_page = SingletonMark('mark_page')
m_cursor = SingletonMark('mark_cursor')


def _replace_page(data, page):
    return nested_map(lambda x: page if x is m_page else x, data)


def _replace_cursor(data, cursor):
    return nested_map(lambda x: cursor if x is m_cursor else x, data)


ReactionTyping = Literal['Like', 'Dislike', 'Heart', 'Laugh', 'Cry']


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

    @classmethod
    def _resp_postprocess(cls, resp):
        try:
            json_ = resp.json()
        except json.JSONDecodeError:
            resp.raise_for_status()
        else:
            if 'error' in json_['result']:
                raise APIError(resp, resp_data_parse(json_['result']['data']))
            else:
                return resp_data_parse(json_['result']['data'])

    def _get(self, url, data=undefined):
        return self._resp_postprocess(self._session.get(
            urljoin(CIVITAI_ROOT, url),
            params={'input': json.dumps(req_data_format(data))},
        ))

    def _post(self, url, data=undefined):
        return self._resp_postprocess(self._session.post(
            urljoin(CIVITAI_ROOT, url),
            json=req_data_format(data),
        ))

    @classmethod
    def _iter_via_cursor_fn(cls, fn):
        cursor = undefined
        while True:
            items, cursor = fn(cursor)
            yield from items

            if cursor is None:
                break

    def _iter_via_cursor(self, url, data, items_key: str = 'items'):
        def _fn(cursor):
            resp_data = self._get(url, _replace_cursor(data, cursor))
            return resp_data[items_key], resp_data['nextCursor']

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

    def get_creator(self, username: str):
        return self._get(
            '/api/trpc/user.getCreator',
            {
                "username": username,
                "authed": self._authed
            }
        )

    def get_creator_by_id(self, userid: int):
        return self._get(
            '/api/trpc/user.getCreator',
            {
                'id': userid,
                'authed': self._authed,
            }
        )

    def get_myself(self):
        return self.get_creator(self._username)

    def get_article(self, article_id: int):
        return self._get(
            '/api/trpc/article.getById',
            {
                "id": article_id,
                "authed": self._authed,
            }
        )

    def get_article_comments(self, article_id: int):
        return self._get(
            '/api/trpc/commentv2.getThreadDetails',
            {
                "entityId": article_id,
                "entityType": "article",
                "hidden": undefined,
                "authed": self._authed,
            }
        )

    def _toggle_reactions(self, entity_id: int, entity_type: str, reaction: ReactionTyping):
        return self._post(
            '/api/trpc/reaction.toggle',
            {
                "entityId": entity_id,
                "entityType": entity_type,
                "reaction": reaction,
                "authed": self._authed
            }
        )

    def toggle_article_engagement(self, article_id: int):
        return self._post(
            '/api/trpc/user.toggleArticleEngagement',
            {
                "type": "Favorite",
                "articleId": article_id,
                "authed": self._authed,
            }
        )

    def toggle_article_reaction(self, article_id: int, reaction: ReactionTyping):
        return self._toggle_reactions(article_id, 'article', reaction)

    def toggle_article_comment_reaction(self, comment_id: int, reaction: ReactionTyping):
        return self._toggle_reactions(comment_id, 'comment', reaction)

    def get_model(self, model_id: int):
        return self._get(
            '/api/trpc/model.getById',
            {
                "id": model_id,
                "authed": self._authed,
            }
        )

    def iter_model_images(self, model_id):
        yield from self._iter_via_cursor(
            '',
            {
                "modelVersionId": model_id,
                "prioritizedUserIds": [self._userid] if self.whoami else [],
                "period": "AllTime",
                "sort": "Most Reactions",
                "limit": 20,
                "browsingMode": "NSFW",
                "cursor": m_cursor,
                "authed": self._authed,
            }
        )

    def get_model_suggested_resources(self, model_id: int):
        return self._get(
            '/api/trpc/model.getAssociatedResourcesCardData',
            {
                "fromId": model_id, "type": "Suggested", "authed": self._authed,
            }
        )

    def get_model_comments(self, model_id: int):
        yield from self._iter_via_cursor(
            '/api/trpc/comment.getAll',
            {
                "modelId": model_id,
                "limit": 8,
                "sort": "newest",
                "hidden": undefined,
                "cursor": m_cursor,
                "authed": self._authed
            },
            items_key='comments',
        )

    def get_model_posts(self, model_id: int, model_version_id: int):
        yield from self._iter_via_cursor(
            '/api/trpc/image.getImagesAsPostsInfinite',
            {
                "period": "AllTime",
                "sort": "Newest",
                "view": "categories",
                "modelVersionId": model_version_id,
                "modelId": model_id,
                "limit": 50,
                "cursor": m_cursor,
                "authed": self._authed,
            }
        )

    def toggle_image_reaction(self, image_id: int, reaction: ReactionTyping):
        return self._toggle_reactions(image_id, 'image', reaction)

    def toggle_model_comment_reaction(self, comment_id: int, reaction: ReactionTyping):
        return self._post(
            '/api/trpc/comment.toggleReaction',
            {
                "id": comment_id,
                "reaction": reaction,
                "authed": self._authed,
            }
        )

    def get_post(self, post_id):
        return self._get(
            '/api/trpc/post.get',
            {
                "id": post_id,
                "authed": self._authed
            }
        )

    def iter_post_images(self, post_id):
        yield from self._iter_via_cursor(
            '/api/trpc/image.getInfinite',
            {
                "postId": post_id,
                "browsingMode": "NSFW",
                "cursor": m_cursor,
                "authed": self._authed,
            }
        )

    def iter_images_by_category(self):
        yield from self._iter_via_cursor(
            '/api/trpc/image.getImagesByCategory',
            {
                "period": "Week",
                "sort": "Newest",
                "view": "categories",
                "limit": 6,
                "browsingMode": "NSFW",
                "cursor": m_cursor,
                "authed": self._authed,
            }
        )
