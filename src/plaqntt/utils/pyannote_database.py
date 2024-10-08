# Copyright (c) 2024 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved
# By Alexis Plaquet, 2024


def protocol_fullname_to_name_subset(extended_name, default_subset) -> tuple[str, str]:
    splitted_pname = extended_name.split(".")
    if len(splitted_pname) == 4:
        return ".".join(splitted_pname[:-1]), splitted_pname[-1]
    elif len(splitted_pname) == 3:
        return extended_name, default_subset
    else:
        raise ValueError(f"Invalid protocol name {extended_name}")
