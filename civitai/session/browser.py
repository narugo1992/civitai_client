from __future__ import annotations

import logging
import random
import time
from typing import Any, Mapping, Tuple
from urllib.parse import quote
from urllib.request import getproxies

from hbutils.string import plural_word
from hbutils.system import urlsplit

from .config import CIVITAI_ROOT

try:
    from selenium import webdriver
    from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
    from webdriver_manager.chrome import ChromeDriverManager
except (ModuleNotFoundError, ImportError):
    _has_selenium = False
else:
    _has_selenium = True

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 " \
             "(KHTML, like Gecko) Chrome/101.0.4951.54 Safari/537.36"
PROXIES = getproxies()


class CivitAIBrowser:
    def __init__(self, headless: bool = False):
        if not _has_selenium:
            raise SystemError('Selenium or webdriver_manager not installed. '
                              'Please install them with requirements-selenium.txt')

        self.caps = DesiredCapabilities.CHROME.copy()
        self.caps["goog:loggingPrefs"] = {
            "performance": "ALL"
        }  # enable performance logs

        self.__browser = webdriver.Chrome(
            executable_path=ChromeDriverManager().install(),
            options=self.__get_chrome_option(headless=headless),
            desired_capabilities=self.caps,
        )
        self.__closed = False

    @staticmethod
    def __get_chrome_option(headless: bool):
        options = webdriver.ChromeOptions()

        if headless:
            options.add_argument("--headless")
            options.add_argument("--disable-gpu")
            options.add_argument("--disable-extensions")
            options.add_argument("--disable-infobars")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--disable-browser-side-navigation")
            options.add_argument("--start-maximized")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--user-agent=" + USER_AGENT)
            options.add_experimental_option("excludeSwitches", ["enable-automation"])
            options.add_experimental_option("useAutomationExtension", False)

        if "all" in PROXIES:
            options.add_argument(f"--proxy-server={PROXIES['all']}")
        elif "https" in PROXIES:
            options.add_argument(f"--proxy-server={PROXIES['https']}")
        elif "http" in PROXIES:
            options.add_argument(f"--proxy-server={PROXIES['http']}")
        else:
            options.add_argument('--proxy-server="direct://"')
            options.add_argument("--proxy-bypass-list=*")

        return options

    def close(self):
        if not self.__closed:
            self.__browser.close()
            self.__closed = True

    def get_civitai_cookie(self, timeout: int = 120):
        from .whoami import _get_whoami_by_page_source

        redirect_url = f'{CIVITAI_ROOT}/'
        login_url = f'{CIVITAI_ROOT}/login?returnUrl={quote(redirect_url)}'

        self.__browser.get(login_url)
        logging.info(f'Please login civitai within {plural_word(timeout, "second")}.')

        for _ in range(timeout):
            logging.info(f'Current url of browser: {self.__browser.current_url!r}')

            if urlsplit(self.__browser.current_url).host == urlsplit(CIVITAI_ROOT).host:
                _whoami = _get_whoami_by_page_source(self.__browser.page_source)
                if _whoami is not None:
                    logging.info(f'Login success! '
                                 f'Hello, @{_whoami.username} (ID: {_whoami.id}, email: {_whoami.email})!')
                    break
            time.sleep(1.0)
        else:
            self.close()
            raise ValueError(f'Login timeout, {plural_word(timeout, "second")} exceed.')

        items = sorted([
            (item['name'], item['value'])
            for item in self.__browser.get_cookies()
            # if item['domain'].endswith('civitai.com')
        ])
        cookies = {key: value for key, value in items}
        return cookies, _whoami.raw

    @staticmethod
    def __sleep_uniform(min_sleep: float, max_sleep: float, slow: bool = True) -> None:
        if slow:
            time.sleep(random.uniform(min_sleep, max_sleep))

    @staticmethod
    def __type_content(elm: Any, text: str, slow: bool = False) -> None:
        if slow:
            for character in text:
                elm.send_keys(character)
                time.sleep(random.uniform(0.3, 0.7))
        else:
            elm.send_keys(text)


def get_civitai_cookies(login_timeout: int = 120) -> Tuple[Mapping[str, str], Mapping[str, Any]]:
    return CivitAIBrowser().get_civitai_cookie(timeout=login_timeout)
