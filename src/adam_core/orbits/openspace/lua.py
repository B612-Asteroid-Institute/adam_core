import json
from enum import Enum
from numbers import Number


def _render_lua_payload(payload: dict, indent: int = 0) -> str:
    from adam_core import _rust_native as _rn

    return _rn.openspace_lua_to_string(json.dumps(payload), indent)


class LuaDict:
    """
    A base class for all Lua types designed to handle formatting
    to a Lua string.
    """

    def to_pascal_case(self, s: str):
        return "".join(word.capitalize() for word in s.split("_"))

    def _to_rust_payload(self):
        fields = []
        for k, v in self.__dict__.items():
            if v is None:
                continue
            fields.append({"key": k, "value": _value_to_rust_payload(v)})
        return {"kind": "dict", "fields": fields}

    def to_string(self, indent: int = 0):
        return _render_lua_payload(self._to_rust_payload(), indent)


def _value_to_rust_payload(v):
    if isinstance(v, bool):
        return {"kind": "bool", "value": v}
    if isinstance(v, Enum):
        return {"kind": "string", "value": v.value}
    if isinstance(v, tuple):
        return {"kind": "tuple", "values": [str(x) for x in v]}
    if isinstance(v, list):
        return {"kind": "list", "values": [str(x) for x in v]}
    if isinstance(v, str):
        return {"kind": "string", "value": v}
    if isinstance(v, LuaDict):
        return v._to_rust_payload()
    if isinstance(v, Number):
        return {"kind": "raw", "value": str(v)}
    return {"kind": "raw", "value": str(v)}
