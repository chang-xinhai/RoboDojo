"""Resolve configured contact namespaces to one concrete environment."""

import re


def contact_path_for_env(path: str, env_index: int) -> str:
    """Expand supported env templates, rejecting unresolved match expressions.

    An absolute global path is shared deliberately; a path in the env namespace
    must target this environment after expansion, never another environment.
    """
    if not isinstance(path, str):
        raise TypeError("TacSL contact paths must be strings")
    result = path.replace("{ENV_REGEX_NS}", f"/World/envs/env_{env_index}")
    for pattern in ("env_[^/]+", "env_.*", "env_.+", "env_0"):
        result = result.replace(f"/World/envs/{pattern}/", f"/World/envs/env_{env_index}/")
    if not result.startswith("/") or any(char in result for char in "{}[]*+?^$()|\\."):
        raise ValueError(f"TacSL requires a concrete absolute contact path after env expansion: {path!r}")
    match = re.match(r"/World/envs/env_(\d+)(?:/|$)", result)
    if match and int(match.group(1)) != env_index:
        raise ValueError(f"TacSL contact path targets another environment: {result}, expected env_{env_index}")
    return result
