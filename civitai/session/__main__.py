import logging
from functools import partial

import click

from ..session import get_civitai_cookies
from ..utils import GLOBAL_CONTEXT_SETTINGS, save_cookies
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
@click.option('-T', '--timeout', 'timeout', type=int, default=60,
              help='Default timeout of login.', show_default=True)
def login(output_file: str, timeout: int):
    logging.basicConfig(level=logging.INFO)
    save_cookies(get_civitai_cookies(timeout), output_file)
    logging.info(f'The cookies file is saved as {output_file!r}.')


if __name__ == '__main__':
    cli()
