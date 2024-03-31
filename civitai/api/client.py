from urllib.parse import urljoin

import requests

from ..session import CIVITAI_ROOT
from ..utils import get_session


class CivitAIAPIClient:
    def __init__(self, session: requests.Session):
        self._session = session

    @classmethod
    def no_login(cls) -> 'CivitAIAPIClient':
        return cls(get_session())

    def _get(self, url):
        resp = self._session.get(urljoin(CIVITAI_ROOT, url))
        resp.raise_for_status()
        return resp.json()

    def get_model_version(self, model_version_id):
        return self._get(f'/api/v1/model-versions/{model_version_id}')

    def get_model_version_by_hash(self, model_hash: str):
        return self._get(f'/api/v1/model-versions/by-hash/{model_hash}')
