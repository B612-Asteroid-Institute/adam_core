from enum import Enum


class LuaDict:
    """
    A base class for all Lua types designed to handle formatting
    to a Lua string.
    """

    def to_pascal_case(self, s: str):
        return "".join(word.capitalize() for word in s.split("_"))

    def to_string(self, indent: int = 0):
        lua = "{\n"
        for k, v in self.__dict__.items():
            if v is None:
                continue
            if isinstance(v, bool):
                v = str(v).lower()
            elif isinstance(v, Enum):
                v = f'"{v.value}"'
            elif isinstance(v, tuple):
                v = f"{{{', '.join([str(x) for x in v])}}}"
            elif isinstance(v, list):
                v = f'{{"{", ".join([str(x) for x in v])}"}}'
            elif isinstance(v, str):
                v = f'"{v}"'
            elif isinstance(v, LuaDict):
                v = v.to_string(indent=indent + 4)

            if k != "gui":
                k_str = self.to_pascal_case(k)
            else:
                k_str = "GUI"

            lua += f"{' ' * indent}{k_str} = {v},\n"

        lua += f"{' ' * (indent - 4)}}}"
        return lua
