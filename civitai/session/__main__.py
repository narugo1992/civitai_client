import logging
from functools import partial

import click

from .browser import get_civitai_cookies
from ..utils import GLOBAL_CONTEXT_SETTINGS, save_cookies, push_cookies_to_hf, get_hf_fs
from ..utils import print_version as _origin_print_version

print_version = partial(_origin_print_version, 'civitai')


@click.group(context_settings={**GLOBAL_CONTEXT_SETTINGS})
@click.option('-v', '--version', is_flag=True,
              callback=print_version, expose_value=False, is_eager=True,
              help="Utils for civitai session session.")
def cli():
    pass  # pragma: no cover


@cli.command('login', context_settings={**GLOBAL_CONTEXT_SETTINGS},
             help='Login civitai and save the cookies file.')
@click.option('-o', '--output_file', 'output_file', type=str, required=True,
              help='Where to save the cookies file.', show_default=True)
@click.option('-T', '--timeout', 'timeout', type=int, default=120,
              help='Default timeout of login.', show_default=True)
def login(output_file: str, timeout: int):
    logging.basicConfig(level=logging.INFO)
    cookies, raw_user_info = get_civitai_cookies(timeout)
    save_cookies(
        cookies=cookies,
        raw_user_info=raw_user_info,
        file=output_file,
    )
    logging.info(f'The cookies file is saved as {output_file!r}.')


@cli.command('login_hf', context_settings={**GLOBAL_CONTEXT_SETTINGS},
             help='Login civitai and push the cookies file to huggingface private repository.')
@click.option('-T', '--timeout', 'timeout', type=int, default=120,
              help='Default timeout of login.', show_default=True)
@click.option('-r', '--repository', 'repository', type=str, required=True,
              help='Repository to save the cookies file.', show_default=True)
def login_hf(timeout: int, repository: str):
    logging.basicConfig(level=logging.INFO)
    logging.info(f'Preparing cookie files.')
    cookies, raw_user_info = get_civitai_cookies(timeout)
    push_cookies_to_hf(
        cookies=cookies,
        raw_user_info=raw_user_info,
        repository=repository,
        file='civitai_cookies.json',
    )

    remote_file = f'datasets/{repository}/civitai_cookies.json'
    hf_fs = get_hf_fs()
    assert hf_fs.exists(remote_file), (f'Remote file {remote_file!r} not found, some error occurred. '
                                       f'Please notify the developer.')
    logging.info(f'Cookie file saved to {remote_file!r}, you can use this string to load the civitai session.')


if __name__ == '__main__':
    cli()
