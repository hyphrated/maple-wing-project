"""Async helpers for interrupting an in-progress model response."""

from __future__ import annotations

import asyncio
from contextlib import suppress


class InterruptionController:
    """Own and cancel the one Ollama generation that is currently running.

    RealtimeSTT can call its voice-activity callbacks from a worker thread, so
    :meth:`request_interrupt` schedules cancellation on the main asyncio loop
    instead of touching an asyncio task directly from that worker thread.
    """

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._active_task: asyncio.Task[str] | None = None
        self._active_prompt: str | None = None
        self._interrupted_prompt: str | None = None

    def bind_to_current_loop(self) -> None:
        """Remember the running event loop used by model-generation tasks."""

        self._loop = asyncio.get_running_loop()

    def track_generation(self, task: asyncio.Task[str], prompt: str) -> None:
        """Register a new model task and the user request that started it."""

        self._active_task = task
        self._active_prompt = prompt

    def forget_generation(self, task: asyncio.Task[str]) -> None:
        """Clear a completed task without erasing interruption context."""

        if task is self._active_task:
            self._active_task = None
            self._active_prompt = None

    def request_interrupt(self) -> None:
        """Request cancellation safely from either asyncio or audio threads."""

        if self._loop is None or self._loop.is_closed():
            return
        self._loop.call_soon_threadsafe(self._cancel_active_task, True)

    def _cancel_active_task(self, remember_prompt: bool) -> None:
        """Cancel the active task on its owning event-loop thread."""

        task = self._active_task
        if task is None or task.done():
            return

        if remember_prompt and self._active_prompt:
            self._interrupted_prompt = self._active_prompt
        task.cancel()

    async def cancel_active_generation(self, remember_prompt: bool = True) -> None:
        """Cancel and await the active generation so its stream closes cleanly."""

        task = self._active_task
        if task is None:
            return

        self._cancel_active_task(remember_prompt)
        with suppress(asyncio.CancelledError):
            await task
        self.forget_generation(task)

    def take_interrupted_prompt(self) -> str | None:
        """Return interruption context once, for use by the replacement turn."""

        prompt = self._interrupted_prompt
        self._interrupted_prompt = None
        return prompt


def contextualize_interruption(latest_prompt: str, interrupted_prompt: str | None) -> str:
    """Tell the model that the latest message refines an interrupted request.

    The earlier user message is already present in conversation history. This
    short note makes the relationship explicit without duplicating that message.
    """

    if not interrupted_prompt:
        return latest_prompt

    return (
        f"{latest_prompt}\n\n"
        "[This message interrupted my previous request. Use any relevant context "
        "from that request, but prioritize what I am asking now.]"
    )


__all__ = ["InterruptionController", "contextualize_interruption"]
