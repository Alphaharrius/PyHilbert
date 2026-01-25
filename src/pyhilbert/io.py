import os
from typing import Set, Optional, Union

from .logging import get_logger


_logger = get_logger(__name__)


_io_dir: Optional[str] = None


def iodir(path: Optional[Union[str, os.PathLike[str]]] = None) -> str:
    """
    Get or set the base directory for IO storage.

    If a path is provided, it becomes the active IO directory. The directory
    is created if needed. If no path is provided, the current IO directory
    is returned; when unset, defaults to ".pickle".

    Parameters
    ----------
    `path`
        Optional filesystem path to use as the IO root.

    Returns
    -------
    `str`
        The resolved IO directory path.
    """
    global _io_dir
    if path is not None:
        _io_dir = os.fspath(path)
        _logger.debug("IO directory set to: %s", _io_dir)
    dir = _io_dir or ".pickle"
    os.makedirs(dir, exist_ok=True)
    _logger.debug("IO directory resolved to: %s", dir)
    return dir


_all_env: Optional[Set[str]] = None
_current_env: Optional[str] = None


def env(name: Optional[str] = None) -> Optional[str]:
    """
    Get or set the active environment name under the IO directory.

    When called without a name, returns the currently active environment if
    one was set during this process, otherwise raises a RuntimeError.
    When a name is provided, ensures a subdirectory exists under the IO
    directory and sets it as the active environment.

    Parameters
    ----------
    `name`
        Environment name to activate, or `None` to query the current one.

    Returns
    -------
    `str`
        The current or newly-set environment name.

    Raises
    ------
    `RuntimeError`
        If no environment is set and `name` is `None`.
    """
    global _all_env
    global _current_env
    if name is None:
        if _current_env is not None:
            return _current_env
        raise RuntimeError("No environment is currently set.")

    if _all_env is None:
        root = iodir()
        _all_env = {
            entry
            for entry in os.listdir(root)
            if os.path.isdir(os.path.join(root, entry))
        }

    if name not in _all_env:
        os.makedirs(os.path.join(iodir(), name), exist_ok=True)
        _all_env.add(name)
        _logger.debug("Environment created: %s", name)

    _current_env = name
    _logger.debug("Environment set to: %s", _current_env)
    return _current_env
