"""Pytest helpers for running asyncio tests without external plugins."""

from __future__ import annotations

import asyncio
import inspect
from typing import Any

import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem: pytest.Function) -> bool | None:
    """Execute tests marked with ``pytest.mark.asyncio`` using an event loop.

    The sandbox used for the kata does not ship with the ``pytest-asyncio``
    plugin, but the provided tests rely on the ``@pytest.mark.asyncio`` marker.
    This hook reproduces the minimal behaviour we need by creating an isolated
    event loop for each async test function.
    """

    marker = pyfuncitem.get_closest_marker("asyncio")
    if not marker:
        return None

    test_function = pyfuncitem.obj
    if not inspect.iscoroutinefunction(test_function):
        return None

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(test_function(**pyfuncitem.funcargs))
    finally:
        loop.close()

    return True


@pytest.fixture
def event_loop() -> Any:
    """Provide a per-test event loop fixture compatible with pytest-asyncio."""

    loop = asyncio.new_event_loop()
    try:
        yield loop
    finally:
        loop.close()
