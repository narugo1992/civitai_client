import datetime
import json
import logging
import os
import re
import textwrap
import uuid
import warnings
from enum import IntFlag
from typing import Optional, List, Union, Tuple, Mapping, Any
from urllib.parse import urljoin

import blurhash
import markdown2
import numpy as np
import requests
from hbutils.design import SingletonMark
from hbutils.string import plural_word
from imgutils.data import load_image
from imgutils.sd import get_sdmeta_from_image
from tqdm import tqdm
from urlobject import URLObject

from .exceptions import SessionError, APIError
from .image import CivitaiImage
from .superjs import resp_data_parse, req_data_format, undefined
from ..session import load_civitai_session, WhoAmI, CIVITAI_ROOT, get_whoami_by_raw_user_info
from ..utils import CookiesTyping, parse_publish_at

try:
    from typing import Literal
except (ImportError, ModuleNotFoundError):
    from typing_extensions import Literal

m_page = SingletonMark('mark_page')
m_cursor = SingletonMark('mark_cursor')
m_none = SingletonMark('mark_none')

PeriodTyping = Literal['Day', 'Week', 'Month', 'Year', 'AllTime']
ImageSortTyping = Literal['Newest', 'Oldest', 'Most Reactions', 'Most Buzz', 'Most Comments', 'Most Collected']
TagSortTyping = Literal['Most Models', 'Most Images', 'Most Posts', 'Most Articles', 'Most Hidden']
ModelSortTyping = Literal[
    'Newest', 'Oldest', 'Most Buzz', 'Most Collected', 'Highest Rated',
    'Most Downloaded', 'Most Liked', 'Most Discussed', 'Most Images',
]


class Level(IntFlag):
    PG = 0x1
    PG13 = 0x2
    R = 0x4
    X = 0x8
    XXX = 0x10

    SFW = PG | PG13
    NSFW = R | X | XXX
    ALL = PG | PG13 | R | X | XXX


def _replace_mark(data, mark, v_mark):
    if isinstance(data, dict):
        retval = {}
        for key, value in data.items():
            if value is not m_none:
                retval[key] = _replace_mark(value, mark, v_mark)
        return type(data)(retval)

    elif isinstance(data, (list, tuple)):
        retval = []
        for value in data:
            if value is not m_none:
                retval.append(_replace_mark(value, mark, v_mark))
        return type(data)(retval)

    else:
        return data if data is not mark else v_mark


def _replace_page(data, page):
    return _replace_mark(data, m_page, page)


def _replace_cursor(data, cursor):
    return _replace_mark(data, m_cursor, cursor)


def _norm(x, keep_space: bool = True):
    return re.sub(r'[\W_]+', ' ' if keep_space else '', x.lower()).strip()


def _model_tag_same(x, y):
    return _norm(x, keep_space=True) == _norm(y, keep_space=True)


def _vae_model_same(x, y):
    return _norm(x, keep_space=False) == _norm(y, keep_space=False)


ReactionTyping = Literal['Like', 'Dislike', 'Heart', 'Laugh', 'Cry']
CommercialUseTyping = Literal['Image', 'RentCivit', 'Rent', 'Sell']
DEFAULT_COMMERCIAL_USE: List[CommercialUseTyping] = ['RentCivit', 'Rent']

ModelTypeTyping = Literal[
    'Checkpoint', 'Embedding', 'Hypernetwork', 'AestheticGradient', 'LORA', 'LoCon', 'DoRA',
    'Controlnet', 'Upscaler', 'MotionModule', 'VAE', 'Poses', 'Wildcards', 'Workflows', 'Other',
]  # attention : LoCon is named as LyCORIS on civitai page
CheckpointTypeTyping = Literal['Trained', 'Merge']


class CivitAIClient:
    def __init__(self, session: requests.Session, raw_user_info: Optional[Mapping[str, Any]] = None):

        self._session = session
        self._whoami_value = get_whoami_by_raw_user_info(raw_user_info)

    @classmethod
    def load(cls, anything: Optional[CookiesTyping] = None) -> 'CivitAIClient':
        session, raw_user_info = load_civitai_session(anything)
        return cls(session=session, raw_user_info=raw_user_info)

    @classmethod
    def no_login(cls) -> 'CivitAIClient':
        return cls.load(None)

    @property
    def whoami(self) -> Optional[WhoAmI]:
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
    def _resp_postprocess(cls, resp, parse: bool = True):
        try:
            json_ = resp.json()
        except json.JSONDecodeError:
            resp.raise_for_status()
        else:
            if 'error' in json_:
                raise APIError(resp, resp_data_parse(json_['error']))
            if 'error' in json_['result']:
                raise APIError(resp, resp_data_parse(json_['result']['data']))

            if parse:
                return resp_data_parse(json_['result']['data'])
            else:
                return json_

    def _get(self, url, data=undefined, parse: bool = True):
        data = req_data_format(data)
        logging.debug(f'GET {url!r}, data: {data!r} ...')
        return self._resp_postprocess(self._session.get(
            urljoin(CIVITAI_ROOT, url),
            params={'input': json.dumps(data)},
        ), parse=parse)

    def _post(self, url, data=undefined, parse: bool = True):
        data = req_data_format(data)
        logging.debug(f'GET {url!r}, data: {data!r} ...')
        return self._resp_postprocess(self._session.post(
            urljoin(CIVITAI_ROOT, url),
            json=data,
        ), parse=parse)

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

    def iter_models(self, username=None, nsfw_level: Level = Level.ALL,
                    sort: ModelSortTyping = 'Newest', period: PeriodTyping = 'AllTime',
                    followed_only: bool = False, generation_supported_only: bool = False,
                    show_early_access: bool = False, types: Optional[List[ModelTypeTyping]] = None):
        yield from self._iter_via_cursor(
            '/api/trpc/model.getAll',
            {
                "period": period,
                "periodMode": "published",
                "sort": sort,
                "earlyAccess": show_early_access,
                "supportsGeneration": generation_supported_only,
                "followed": followed_only,
                "browsingLevel": int(nsfw_level),
                "username": username if username else m_none,
                "types": types if types else m_none,
                "cursor": m_cursor,
                "authed": self._authed,
            }
        )

    def iter_models_of_user(self, username):
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
        yield from self.iter_models_of_user(self._username)

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

    def iter_images(self, username=None, period: Optional[PeriodTyping] = 'AllTime',
                    sort: Optional[ImageSortTyping] = 'Newest', level: Optional[Level] = Level.ALL,
                    post_id: Optional[int] = None, no_type: bool = False):
        params = {
            "postId": post_id if post_id is not None else m_none,
            "period": period if period is not None else m_none,
            "sort": sort if sort is not None else m_none,
            "types": ["image"] if not no_type else m_none,
            "browsingLevel": int(level) if level is not None else None,
            "cursor": m_cursor,
            "authed": self._authed,
        }
        if username:
            params['username'] = username
        yield from self._iter_via_cursor('/api/trpc/image.getInfinite', params)

    def iter_images_self(self):
        yield from self.iter_images(username=self._username, period='AllTime', sort='Newest')

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
        yield from self.iter_images(
            period=None,
            sort=None,
            level=Level.ALL,
            post_id=post_id,
            no_type=True,
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

    def iter_model_tags(self, tag: str):
        return self._iter_via_page(
            '/api/trpc/tag.getAll',
            {
                "limit": 20,
                "entityType": ["Model"],
                "categories": False,
                "query": tag,
                "authed": True,
                'page': m_page,
            }
        )

    def iter_tags(self, entity_type: str, query: str = None,
                  sort: TagSortTyping = None, categories_only: bool = False, votable: bool = False):
        if entity_type == 'Model':
            sort = sort or 'Most Models'
        elif entity_type == 'Image':
            sort = sort or 'Most Images'
        elif entity_type == 'Post':
            sort = sort or 'Most Posts'
        elif entity_type == 'Article':
            sort = sort or 'Most Articles'
        else:
            raise ValueError(f'Unknown entity type - {entity_type!r}.')
        return self._iter_via_page(
            '/api/trpc/tag.getAll',
            {
                "entityType": [entity_type],
                "types": ["UserGenerated", "Label"] if votable else m_none,
                "sort": sort,
                "query": query if query else m_none,
                # "unlisted": False,
                "categories": categories_only,
                "limit": 100,
                "include": ["nsfwLevel"],
                "authed": self._authed,
                "page": m_page,
            }
        )

    def get_votable_tags(self, image_id: int, force_auth: bool = False):
        return self._get(
            '/api/trpc/tag.getVotableTags',
            {
                "id": image_id,
                "type": "image",
                "authed": True if force_auth else self._authed,
            }
        )

    def query_vae_models(self):
        return self._get(
            '/api/trpc/modelVersion.getModelVersionsByModelType',
            {
                "type": "VAE",
                "authed": True,
            }
        )

    def _query_model_tag(self, tag: str) -> Optional[dict]:
        logging.info(f'Querying tag {tag!r} from civitai ...')
        for item in self.iter_model_tags(tag):
            if _model_tag_same(item['name'], tag):
                logging.info(f'Tag {item["name"]}({item["id"]}) found on civitai.')
                return item
        else:
            logging.info(f'Tag not found on civitai, new tag {tag!r} will be created.')
            return None

    def upsert_model(self, name, description_md: str, tags: List[str], category: str = 'character',
                     type_: ModelTypeTyping = 'LORA', checkpoint_type: CheckpointTypeTyping = 'Trained',
                     commercial_use: List[CommercialUseTyping] = None, allow_no_credit: bool = True,
                     allow_derivatives: bool = True, allow_different_licence: bool = True,
                     nsfw: bool = False, poi: bool = False, exist_model_id: Optional[int] = None):
        tags_data, exist_tag_ids, exist_tags = [], set(), set()
        for tag in [*tags, category]:
            tag_item = self._query_model_tag(tag)
            if tag_item:
                if tag_item['id'] not in exist_tag_ids and tag_item['name'].lower() not in exist_tags:
                    if tag_item['isCategory']:
                        tags_data.append({'id': tag_item['id'], 'name': tag_item['name'], 'models': undefined})
                    else:
                        tags_data.append({'id': tag_item['id'], 'name': tag_item['name'], 'isCategory': False})
                    exist_tag_ids.add(tag_item['id'])
                    exist_tags.add(tag_item['name'].lower())

            else:
                if tag.lower() not in exist_tags:
                    tags_data.append({'id': undefined, 'name': tag})
                    exist_tags.add(tag.lower())

        commercial_use = DEFAULT_COMMERCIAL_USE if commercial_use is None else commercial_use
        post_json = {
            "name": name,
            "description": markdown2.markdown(textwrap.dedent(description_md)),
            "type": type_,
            "checkpointType": None if type_.upper() != 'Checkpoint' else checkpoint_type,

            "allowCommercialUse": commercial_use,  # None, Image, Rent, Sell
            "allowNoCredit": allow_no_credit,
            "allowDerivatives": allow_derivatives,
            "allowDifferentLicense": allow_different_licence,

            "nsfw": nsfw,
            "poi": poi,
            "tagsOnModels": tags_data,

            "authed": True,
            "status": "Draft",
            "uploadType": "Created",
            'templateId': undefined,
            'bountyId': undefined,
        }
        if exist_model_id:
            post_json['id'] = exist_model_id
            post_json["locked"] = False
            post_json["status"] = "Published"
            logging.info(f'Model {name!r}({exist_model_id}) already exist, updating its new information. '
                         f'Tags: {[item["name"] for item in tags_data]!r} ...')
        else:
            logging.info(f'Creating model {name!r}, tags: {[item["name"] for item in tags_data]!r} ...')

        return self._post(
            '/api/trpc/model.upsert',
            post_json
        )

    def upsert_version(
            self, model_id: int, version_name: str, description_md: str, trigger_words: List[str],
            base_model: str = 'SD 1.5', steps: Optional[int] = None, epochs: Optional[int] = None,
            clip_skip: Optional[int] = 2, vae_name: Optional[str] = None, early_access_time: int = 0,
            recommended_resources: List[int] = None, require_auth_when_download: bool = False,
            exist_version_id: Optional[int] = None
    ):
        vae_id = None
        if vae_name:
            for vae_item in self.query_vae_models():
                if _vae_model_same(vae_item['modelName'], vae_name):
                    vae_id = vae_item['id']

        logging.info(f'Creating version {version_name!r} for model {model_id}, with base model {base_model!r} ...')
        post_json = {
            "modelId": model_id,
            "name": version_name,
            "baseModel": base_model,
            "baseModelType": undefined,
            "description": markdown2.markdown(textwrap.dedent(description_md)),
            "steps": steps,
            "epochs": epochs,
            "clipSkip": clip_skip,
            "vaeId": vae_id if vae_id is not None else undefined,
            "trainedWords": trigger_words,
            "earlyAccessTimeFrame": early_access_time,
            "skipTrainedWords": bool(not trigger_words),
            "recommendedResources": [
                {'resourceId': rid, 'settings': {}}
                for rid in list(recommended_resources or [])
            ],
            "authed": True,
            "templateId": undefined,
            "bountyId": undefined,
            "requireAuth": require_auth_when_download,
            "monetization": undefined,
        }
        if exist_version_id:
            post_json['id'] = exist_version_id
            post_json["locked"] = False
            post_json["status"] = "Published"
            logging.info(f'Version {version_name!r}({exist_version_id}) already exist in model {model_id!r}, '
                         f'updating its new information ...')
        else:
            logging.info(f'Creating model version {version_name!r}, for model {model_id!r} ...')

        return self._post(
            '/api/trpc/modelVersion.upsert',
            post_json
        )

    def _upload_file(self, local_file: str, type_: str = 'model', filename: str = None):
        filename = filename or os.path.basename(local_file)
        logging.info(f'Creating uploading request for {filename!r} ...')

        resp = self._session.post(
            'https://civitai.com/api/upload',
            json={
                "filename": filename,
                "type": type_,
                "size": os.path.getsize(local_file),
            },
            headers={'Referer': f'https://civitai.com/models/0/wizard?step=3'}
        )
        resp.raise_for_status()
        upload_data = resp.json()
        assert len(upload_data['urls']) == 1, \
            'Multipart uploading not supported yet, please kick narugo1992\'s ass on issue block.'

        logging.info(f'Uploading file {local_file!r} as {filename!r} ...')
        with open(local_file, 'rb') as f:
            resp = self._session.put(
                upload_data['urls'][0]['url'], data=f,
                headers={'Referer': f'https://civitai.com/models/0/wizard?step=3'},
            )
            resp.raise_for_status()
            etag = resp.headers['ETag']

        logging.info(f'Completing uploading for {filename!r} ...')
        resp = self._session.post(
            'https://civitai.com/api/upload/complete',
            json={
                "bucket": upload_data['bucket'],
                "key": upload_data['key'],
                "type": type_,
                "uploadId": upload_data['uploadId'],
                "parts": [{"ETag": etag, "PartNumber": 1}],
            },
            headers={'Referer': f'https://civitai.com/models/0/wizard?step=3'},
        )
        resp.raise_for_status()

        return {
            "url": str(URLObject(upload_data['urls'][0]['url']).without_query()),
            "bucket": upload_data['bucket'],
            "key": upload_data['key'],
            "name": filename,
            "uuid": str(uuid.uuid4()),
            "sizeKB": os.path.getsize(local_file) / 1024.0,
        }

    def upload_models(self, model_version_id: int, model_files: List[Union[str, Tuple[str, str]]]):
        file_items = []
        for file_item in model_files:
            if isinstance(file_item, str):
                local_file, filename = file_item, os.path.basename(file_item)
            elif isinstance(file_item, tuple):
                local_file, filename = file_item
            else:
                raise TypeError(f'Unknown file type - {file_item!r}.')
            file_items.append((local_file, filename))

        for local_file, filename in file_items:
            upload_data = self._upload_file(local_file, 'model', filename)
            logging.info(f'Creating {filename!r} as model file of version {model_version_id} ...')
            self._post(
                '/api/trpc/modelFile.create',
                {
                    **upload_data,
                    "modelVersionId": model_version_id,
                    "type": "Model",
                    "metadata": {
                        "size": None,
                        "fp": None
                    },
                    "authed": True
                },
            )

    @classmethod
    def _get_meta_from_image_file(cls, image_file: str):
        sd_meta = get_sdmeta_from_image(image_file)
        if not sd_meta:
            return {}

        else:
            meta = {
                'prompt': sd_meta.prompt,
                'negativePrompt': sd_meta.neg_prompt,
            }
            for key, value in sd_meta.parameters.items():
                if isinstance(value, tuple):
                    meta[key] = f'{value[0]}x{value[1]}'
                else:
                    meta[key] = value
            if sd_meta.parameters.get('CFG scale'):
                meta['cfgScale'] = int(sd_meta.parameters['CFG scale'])
            if sd_meta.parameters.get('Steps'):
                meta['steps'] = int(sd_meta.parameters['Steps'])
            if sd_meta.parameters.get('Sampler'):
                meta['sampler'] = sd_meta.parameters['Sampler']
            if sd_meta.parameters.get('Seed'):
                meta['seed'] = int(sd_meta.parameters['Seed'])
            if sd_meta.parameters.get('Size'):
                width, height = sd_meta.parameters['Size']
                meta['Size'] = f'{width}x{height}'
            if sd_meta.parameters.get('Clip skip'):
                meta['clipSkip'] = int(sd_meta.parameters['Clip skip'])
            if sd_meta.parameters.get('Model'):
                meta['Model'] = sd_meta.parameters['Model']
            if sd_meta.parameters.get('Model hash'):
                model_hash = sd_meta.parameters['Model hash']
                meta['hashes'] = {'model': model_hash}
                meta["Model hash"] = model_hash
            if sd_meta.parameters.get('Hires resize'):
                meta["Hires resize"] = sd_meta.parameters['Hires resize']
            if sd_meta.parameters.get('Hires steps'):
                meta["Hires steps"] = sd_meta.parameters['Hires steps']
            if sd_meta.parameters.get('Hires upscaler'):
                meta["Hires upscaler"] = sd_meta.parameters['Hires upscaler']
            if sd_meta.parameters.get('Denoising strength'):
                meta["Denoising strength"] = sd_meta.parameters['Denoising strength']
            if sd_meta.parameters.get('Model') and sd_meta.parameters.get('Model hash'):
                meta["resources"] = [
                    {
                        "hash": sd_meta.parameters['Model hash'],
                        "name": sd_meta.parameters['Model'],
                        "type": "model"
                    }
                ]

            return meta

    @classmethod
    def _get_clamped_size(cls, width, height, max_val, _type='all'):
        if _type == 'all':
            if width >= height:
                _type = 'width'
            elif height >= width:
                _type = 'height'

        if _type == 'width' and width > max_val:
            return max_val, int(round((height / width) * max_val))

        if _type == 'height' and height > max_val:
            return int(round((width / height) * max_val)), max_val

        return width, height

    @classmethod
    def _get_image_info(cls, local_file: str) -> Tuple[int, int, str]:
        img = load_image(local_file, force_background='white', mode='RGB')
        new_width, new_height = cls._get_clamped_size(img.width, img.height, 32)
        return img.width, img.height, blurhash.encode(np.array(img.resize((new_width, new_height))))

    def query_post_tags(self, tag: str):
        resp = self._session.get(
            'https://civitai.com/api/trpc/post.getTags',
            params={
                'batch': '1',
                'input': json.dumps({
                    "0": {
                        "json": {
                            "query": tag,
                            "authed": True
                        }
                    }
                })
            }
        )
        resp.raise_for_status()
        return resp.json()[0]['result']['data']['json']

    def upload_image(self, local_file: str) -> CivitaiImage:
        filename = os.path.basename(local_file)
        resp = self._session.post(
            'https://civitai.com/api/image-upload',
            json={
                "filename": filename,
                "metadata": {}
            },
        )
        resp.raise_for_status()
        upload_id = resp.json()['id']
        upload_url = resp.json()['uploadURL']

        logging.info(f'Uploading local image {local_file!r} as image {filename!r} ...')
        with open(local_file, 'rb') as f:
            resp = self._session.put(upload_url, data=f)
            resp.raise_for_status()

        return CivitaiImage(id=upload_id, filename=filename)

    def upload_images_for_model_version(
            self, model_version_id: int,
            image_files: Union[List[str], List[Tuple[str, str]]],
            tags: List[str], nsfw: bool = False,
    ):
        logging.info(f'Creating post for model version {model_version_id} ...')
        resp = self._post(
            '/api/trpc/post.create',
            {
                "modelVersionId": model_version_id,
                "authed": True,
            },
        )
        post_id = resp['id']

        for index, upload_item in enumerate(tqdm(image_files,
                                                 desc=f'Uploading {plural_word(len(image_files), "image")}')):
            if isinstance(upload_item, tuple):
                local_file, filename = upload_item
            elif isinstance(upload_item, str):
                local_file = upload_item
                filename = os.path.basename(local_file)
            else:
                raise TypeError(f'Unknown type of upload image - {upload_item!r}.')

            upload_image = self.upload_image(local_file)

            logging.info(f'Completing the uploading of {filename!r} ...')
            width, height, bhash = self._get_image_info(local_file)
            self._post(
                '/api/trpc/post.addImage',
                {
                    "type": "image",
                    "index": index,
                    "uuid": str(uuid.uuid4()),
                    "name": filename,
                    "meta": self._get_meta_from_image_file(local_file),
                    "url": upload_image.id,
                    "mimeType": "image/png",
                    "hash": bhash,
                    "width": width,
                    "height": height,
                    "status": "uploading",
                    "message": undefined,
                    "postId": post_id,
                    "modelVersionId": model_version_id,
                    "authed": True
                }
            )

        for tag in tags:
            for tag_item in self.query_post_tags(tag):
                if _model_tag_same(tag_item['name'], tag):
                    tag_id, tag_name = tag_item['id'], tag_item['name']
                    break
            else:
                tag_id, tag_name = None, tag

            if tag_id is not None:
                logging.info(f'Adding tag {tag_name!r}({tag_id}) for post {post_id!r} ...')
            else:
                logging.info(f'Creating and adding new tag {tag_name!r} for post {post_id!r} ...')
            try:
                _ = self._post(
                    '/api/trpc/post.addTag',
                    {
                        "id": post_id,
                        "tagId": tag_id if tag_id is not None else undefined,
                        "name": tag_name,
                        "authed": True,
                    },
                )
            except APIError as err:
                # 2024-1-27, this api will sometimes down. I don't know why
                # <APIError status: 404, error: {'message': 'No Post found', 'code': -32004, 'data': {'code': 'NOT_FOUND', 'httpStatus': 404, 'path': 'post.addTag'}}>
                warnings.warn(f'Error occurred when adding tag: {err!r}.')

        logging.info(f'Marking for nsfw ({nsfw!r}) ...')
        _ = self._post(
            '/api/trpc/post.update',
            {
                'id': post_id,
                'nsfw': nsfw,
                'authed': True,
            },
            parse=False,
        )

        return post_id

    def model_publish(self, model_id: int, model_version_id: int, publish_at=None):
        publish_at = parse_publish_at(publish_at)
        if publish_at:
            logging.info(f'Publishing model {model_id!r}\'s version {model_version_id!r}, '
                         f'at {publish_at.isoformat()!r} ...')
        else:
            logging.info(f'Publishing model {model_id!r}\'s version {model_version_id!r} ...')

        return self._post(
            '/api/trpc/model.publish',
            {
                "id": model_id,
                "versionIds": [
                    model_version_id
                ],
                "publishedAt": publish_at if publish_at is not None else undefined,
                "authed": True
            }
        )

    def model_version_publish(self, model_version_id: int, publish_at=None):
        publish_at = parse_publish_at(publish_at)
        if publish_at:
            logging.info(f'Publishing model version {model_version_id!r}, '
                         f'at {publish_at.isoformat()!r} ...')
        else:
            logging.info(f'Publishing model version {model_version_id!r} ...')

        return self._post(
            '/api/trpc/modelVersion.publish',
            {
                "id": model_version_id,
                "publishedAt": publish_at if publish_at is not None else undefined,
                "authed": True
            }
        )

    def model_delete(self, model_id: int):
        return self._post(
            '/api/trpc/model.delete',
            {
                "id": model_id,
                "permanently": False,
                "authed": True
            }
        )

    def model_set_associate(self, model_id: int, resource_ids: List[int]):
        logging.info(f'Setting resource association to model {model_id!r}: {resource_ids!r} ...')
        return self._post(
            '/api/trpc/model.setAssociatedResources',
            {
                "fromId": model_id,
                "type": "Suggested",
                "associations": [
                    {
                        "id": undefined,
                        "resourceType": "model",
                        "resourceId": rid,
                    }
                    for rid in resource_ids
                ],
                "authed": True
            }
        )

    def post_publish(self, post_id: int, publish_at: Optional[str] = None):
        return self._post(
            'https://civitai.com/api/trpc/post.update',
            {
                'id': post_id,
                'authed': True,
                "publishedAt": parse_publish_at(publish_at or datetime.datetime.now()),
            }
        )

    def model_thumb_up(self, model_id: int, model_version_id: int):
        return self._post(
            '/api/trpc/resourceReview.create',
            {
                "modelId": model_id,
                "modelVersionId": model_version_id,
                "recommended": True,
                "rating": 5,
                "authed": self._authed
            }
        )
