# Copyright (c) 2024 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved
# By Alexis Plaquet, 2024
"""Contains types to be used in argparse.
e.g.: `parser.add_argument("--factor", type=nullable_float, default=None, help="Factor to use if any")`
"""

from typing import Callable


def nullable_int(x) -> int | None:
    return None if x.lower() == "none" else int(x)


def nullable_float(x) -> float | None:
    return None if x.lower() == "none" else float(x)


def nullable_float_or_int(x) -> float | int | None:
    return nullable_int(x) if "." not in x else nullable_float(x)


def one_of_list(allowed_values: list[str]) -> Callable[[str], str]:
    def type_fn(x: str) -> str:
        x = x.lower()
        if x in allowed_values:
            return x
        raise ValueError(f"{x} must be one of: " + ", ".join(allowed_values))

    return type_fn


def strtobool(x: str) -> bool:
    if x.lower() in ["true", "1"]:
        return True
    if x.lower() in ["false", "0"]:
        return False
    raise ValueError("Invalid value for bool: " + x)
