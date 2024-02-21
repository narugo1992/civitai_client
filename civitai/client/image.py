from dataclasses import dataclass
from urllib.parse import urljoin, quote_plus

CIVITAI_IMG_ROOT = 'https://image.civitai.com/xG1nkqKTMzGDvpLrqFT7WA/'


@dataclass
class CivitaiImage:
    id: str
    filename: str

    @property
    def original_url(self) -> str:
        return urljoin(CIVITAI_IMG_ROOT, f'{self.id}/original=true/{quote_plus(self.filename)}')

    def get_width_url(self, width: int = 525) -> str:
        return urljoin(CIVITAI_IMG_ROOT, f'{self.id}/width={width}/{quote_plus(self.filename)}')
