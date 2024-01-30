import copy
import datetime
from typing import Dict, Callable, Any, Tuple, Type, Optional

import dateparser

from civitai.utils import parse_time


class _JsUndefined:
    pass


undefined = _JsUndefined()

_SUPERJSON_PARSERS: Dict[str, Callable[[Any], Any]] = {
    'Date': dateparser.parse,
    'undefined': lambda x: None,
}


def resp_data_parse(native):
    raw, meta = native['json'], native.get('meta') or {}
    retval = copy.deepcopy(raw)

    if meta.get('values'):
        if isinstance(meta['values'], dict):
            for key_str, value in meta['values'].items():
                key_segments = key_str.split('.')
                node, key = retval, None
                for segment in key_segments:
                    if key is None:
                        key = segment
                    else:
                        if isinstance(node, list):
                            key = int(key)
                        node, key = node[key], segment

                assert key is not None, f'Empty key string {key_str!r} detected.'
                node[key] = _SUPERJSON_PARSERS[value[0]](node[key])

        elif isinstance(meta['values'], list):
            assert len(meta['values']) == 1, \
                f'Values of meta should have only one element, but {meta["values"]!r} found.'
            retval = _SUPERJSON_PARSERS[meta['values'][0]](retval)

        else:
            raise TypeError(f'Unknown meta values type - {meta["values"]!r}.')

    return retval


def _format_time(publish_at: Optional[str] = None) -> Optional[str]:
    try:
        from zoneinfo import ZoneInfo
    except (ImportError, ModuleNotFoundError):
        from backports.zoneinfo import ZoneInfo

    if publish_at is not None:
        local_time = parse_time(publish_at)
        publish_at = local_time.astimezone(ZoneInfo('UTC')).isoformat()

    return publish_at


_SUPERJSON_FORMATTERS: Dict[Type, Tuple[str, Callable[[Any], Any]]] = {
    _JsUndefined: ('undefined', lambda x: None),
    datetime.datetime: ('Date', _format_time),
}


def req_data_format(native):
    meta = {}

    def _recursion(node, keys):
        if isinstance(node, list):
            return [
                _recursion(item, [*keys, i])
                for i, item in enumerate(node)
            ]
        elif isinstance(node, dict):
            return {
                key: _recursion(value, [*keys, key])
                for key, value in node.items()
            }
        elif type(node) in _SUPERJSON_FORMATTERS:
            _type_name, _type_formatter = _SUPERJSON_FORMATTERS[type(node)]
            nonlocal meta
            if keys:
                meta['.'.join(map(str, keys))] = [_type_name]
            else:
                meta = [_type_name]
            return _type_formatter(node)
        else:
            return node

    retval = _recursion(native, [])

    if meta:
        return {'json': retval, 'meta': {'values': meta}}
    else:
        return {'json': retval}
