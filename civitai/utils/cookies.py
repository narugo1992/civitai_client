import collections.abc
import json
import os.path
import pickle
import warnings
from binascii import Error
from typing import Mapping, Union, Any

from hbutils.encoding import base64_encode, base64_decode
from hbutils.system import TemporaryDirectory

from .huggingface import get_hf_fs, get_hf_client


def encode_cookies(cookies: Mapping[str, str]) -> str:
    return base64_encode(pickle.dumps(cookies), urlsafe=True)


def decode_cookies(b64: str) -> Mapping[str, str]:
    return pickle.loads(base64_decode(b64, urlsafe=True))


hf_fs = get_hf_fs()
hf_client = get_hf_client()

CookiesTyping = Union[str, Mapping[str, str]]


def _load_raw_cookies(anything: CookiesTyping):
    if isinstance(anything, str) and os.path.exists(anything):
        with open(anything, 'r', encoding='utf-8') as f:
            return json.load(f)
    elif isinstance(anything, collections.abc.Mapping):
        return anything
    elif isinstance(anything, str):
        try:
            return decode_cookies(anything)
        except (Error, pickle.UnpicklingError):
            try:
                if hf_fs.exists(anything):
                    return json.loads(hf_fs.read_text(anything))
            except NotImplementedError:
                pass

    raise TypeError(f'Unknown cookie type - {anything!r}.')


def load_cookies(anything: CookiesTyping):
    raw = _load_raw_cookies(anything)
    if 'cookies' in raw:  # new cookies file
        return raw['cookies'], raw['raw_user_info']
    else:
        return raw, None


def save_cookies(cookies: Mapping[str, str], raw_user_info: Mapping[str, Any], file: str):
    with open(file, 'w', encoding='utf-8') as f:
        json.dump({
            'cookies': cookies,
            'raw_user_info': raw_user_info,
        }, f, ensure_ascii=False, indent=4, sort_keys=True)


def push_cookies_to_hf(cookies: Mapping[str, str], raw_user_info: Mapping[str, Any],
                       repository: str, file: str = 'civitai_cookies.json',
                       revision: str = 'main', repo_type: str = 'dataset', public: bool = False):
    hf_client.create_repo(repo_id=repository, repo_type=repo_type, private=not public, exist_ok=True)
    if not hf_client.repo_info(repo_id=repository, repo_type=repo_type).private:
        warnings.warn('Cookies is sensitive information for your civitai account. '
                      'Push it to public repository is dangerous. '
                      'We strongly recommend you to push it to your private repository.')
    with TemporaryDirectory() as td:
        cookie_file = os.path.join(td, 'cookies.json')
        save_cookies(
            cookies=cookies,
            raw_user_info=raw_user_info,
            file=cookie_file
        )
        hf_client.upload_file(
            path_or_fileobj=cookie_file,
            path_in_repo=file,
            repo_id=repository,
            repo_type=repo_type,
            revision=revision,
            commit_message=f'Publish civitai cookies to {file!r}'
        )
