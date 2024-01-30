import datetime
from typing import Optional

import dateparser

try:
    from zoneinfo import ZoneInfo
except (ImportError, ModuleNotFoundError):
    from backports.zoneinfo import ZoneInfo


def parse_time(time):
    if isinstance(time, str):
        d = dateparser.parse(time)
    elif isinstance(time, (int, float)):
        d = datetime.datetime.fromtimestamp(time)
    elif isinstance(time, datetime.datetime):
        d = time
    else:
        raise TypeError(f'Unknown time type - {time}.')

    if not d.tzinfo:
        d = d.astimezone()

    return d


def parse_publish_at(publish_at: Optional[str] = None) -> Optional[datetime.datetime]:
    if publish_at is not None:
        local_time = parse_time(publish_at)
        publish_at = local_time.astimezone(ZoneInfo('UTC'))

    return publish_at
