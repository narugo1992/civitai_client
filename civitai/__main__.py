import json
import logging
import os.path
import time
from contextlib import closing
from functools import partial
from tempfile import TemporaryDirectory

import click
from huggingface_hub import HfApi

from .utils import GLOBAL_CONTEXT_SETTINGS
from .utils import print_version as _origin_print_version
from .session import CivitAIBrowser

print_version = partial(_origin_print_version, 'civitai')


@click.group(context_settings={**GLOBAL_CONTEXT_SETTINGS})
@click.option('-v', '--version', is_flag=True,
              callback=print_version, expose_value=False, is_eager=True,
              help="Utils for civitai session session.")
def cli():
    pass  # pragma: no cover


@cli.command('deploy', context_settings={**GLOBAL_CONTEXT_SETTINGS},
             help='Deploy session data to huggingface.')
@click.option('-r', '--repository', 'repository', type=str, default='narugo/civitai_session',
              help='Repository to deploy session data.', show_default=True)
def deploy(repository: str):
    logging.basicConfig(level=logging.INFO)

    with TemporaryDirectory() as td:
        browser = CivitAIBrowser()
        with closing(browser):
            cookies = browser.get_civitai_cookie()

        output_file = os.path.join(td, 'session.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'cookies': cookies,
                'timestamp': time.time(),
            }, f, indent=4, ensure_ascii=False)

        hf_client = HfApi(token=os.environ['HF_TOKEN'])
        hf_client.create_repo(repo_id=repository, repo_type='dataset', private=True, exist_ok=True)

        hf_client.upload_file(
            repo_id=repository,
            repo_type='dataset',
            path_or_fileobj=output_file,
            path_in_repo='session.json',
        )


if __name__ == '__main__':
    cli()
