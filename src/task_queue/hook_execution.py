"""Shared failure policy for sequential result hook execution."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable, Iterable


class SequentialHookExecutor:
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    async def run(
        self,
        hooks: Iterable[Any],
        *,
        phase: str,
        invoke: Callable[[Any], Awaitable[None]],
        raise_critical: bool,
    ) -> None:
        for hook in hooks:
            try:
                await invoke(hook)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                should_raise = bool(hook.critical) and raise_critical
                log = self.logger.error if should_raise else self.logger.warning
                log(
                    "Hook execution failed: phase=%s hook=%s error_type=%s",
                    phase,
                    hook.name,
                    type(exc).__name__,
                )
                self.logger.debug(
                    "Hook execution failure details: phase=%s hook=%s",
                    phase,
                    hook.name,
                    exc_info=(type(exc), exc, exc.__traceback__),
                )
                if should_raise:
                    raise


__all__ = ["SequentialHookExecutor"]
