"""Low-latency text and speech input used by Fern's conversation loop."""

from __future__ import annotations

import asyncio
import logging
import re
import select
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Literal

from stt_interrupt import InterruptionController, contextualize_interruption

try:
    from RealtimeSTT import AudioToTextRecorder
except ImportError:  # Let typed input continue when voice extras are not installed.
    AudioToTextRecorder = None

InputSource = Literal["text", "voice"]


class _TerminalController:
    """Keep asynchronous output separate from the active ``You: `` input line.

    Windows console input is read one character at a time so the current text can
    be redrawn after Fern or voice recognition prints. The polling loops also let
    the application shut down without leaving a blocked input thread behind.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._stop_requested = threading.Event()
        self._prompt = "You: "
        self._buffer: list[str] = []
        self._input_active = False

    def _write(self, text: str) -> None:
        """Write and immediately flush terminal text while holding the lock."""

        sys.stdout.write(text)
        sys.stdout.flush()

    def _clear_input_line(self) -> None:
        """Erase the visible prompt and tracked input before background output."""

        visible_width = len(self._prompt) + len(self._buffer)
        self._write("\r" + (" " * visible_width) + "\r")

    def print_line(self, message: str) -> None:
        """Print a complete line and restore any input currently being entered."""

        with self._lock:
            if self._input_active:
                self._clear_input_line()

            self._write(message.rstrip("\n") + "\n")

            if self._input_active:
                self._write(self._prompt + "".join(self._buffer))

    def read_line(self) -> str:
        """Read one prompt while allowing asynchronous output to redraw it."""

        if sys.platform == "win32":
            return self._read_windows_line()
        return self._read_portable_line()

    def _begin_input(self) -> None:
        """Initialize and render a fresh user-input line."""

        with self._lock:
            self._buffer.clear()
            self._input_active = True
            self._write(self._prompt)

    def _finish_input(self) -> str:
        """Finish the active line and return the text accumulated within it."""

        with self._lock:
            text = "".join(self._buffer)
            self._buffer.clear()
            self._input_active = False
            self._write("\n")
            return text

    def _read_windows_line(self) -> str:
        """Read Windows console keys while retaining the editable input buffer."""

        import msvcrt

        self._begin_input()
        while not self._stop_requested.is_set():
            if not msvcrt.kbhit():
                time.sleep(0.01)
                continue

            character = msvcrt.getwch()
            if character in {"\x00", "\xe0"}:
                # Consume the second byte of arrows/function keys. A full line
                # editor is unnecessary here; normal text and backspace suffice.
                if msvcrt.kbhit():
                    msvcrt.getwch()
                continue
            if character == "\r":
                return self._finish_input()
            if character == "\x03":
                self._finish_input()
                return "exit"
            if character == "\b":
                with self._lock:
                    if self._buffer:
                        self._buffer.pop()
                        self._write("\b \b")
                continue
            if character.isprintable() or character == "\t":
                with self._lock:
                    self._buffer.append(character)
                    self._write(character)

        return self._finish_input()

    def _read_portable_line(self) -> str:
        """Provide a stoppable line reader for non-Windows terminals."""

        self._begin_input()
        while not self._stop_requested.is_set():
            readable, _, _ = select.select([sys.stdin], [], [], 0.05)
            if readable:
                line = sys.stdin.readline()
                with self._lock:
                    self._buffer[:] = list(line.rstrip("\r\n"))
                return self._finish_input()
        return self._finish_input()

    def stop(self) -> None:
        """Signal an active input poll to return during application shutdown."""

        self._stop_requested.set()


_TERMINAL = _TerminalController()


def terminal_print(message: str) -> None:
    """Print model, status, or voice text without overwriting ``You: ``."""

    _TERMINAL.print_line(message)


def stop_terminal_input() -> None:
    """Release the terminal input worker when the assistant exits."""

    _TERMINAL.stop()


@dataclass
class PromptRequest:
    """A user message passed from an input producer to the model loop."""

    text: str
    source: InputSource
    accepted: asyncio.Event = field(default_factory=asyncio.Event)


# Only stand-alone hesitation sounds are removed. Words containing these
# sequences, such as "umbrella" or "thumb", remain untouched.
_FILLER_WORD_PATTERN = re.compile(
    r"(?<!\w)(?:u+m+|u+h+|e+r+m*|h+m+)(?!\w)\s*[,;:.-]*\s*",
    flags=re.IGNORECASE,
)


def remove_filler_words(transcript: str) -> str:
    """Remove common hesitation sounds and repair the surrounding spacing."""

    cleaned = _FILLER_WORD_PATTERN.sub(" ", transcript)
    cleaned = re.sub(r"\s+([,.!?])", r"\1", cleaned)
    cleaned = re.sub(r"^[,;:.-]+\s*", "", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()


class SpeechToTextListener:
    """Capture microphone turns and place cleaned transcripts on a prompt queue.

    A tiny English model, one-beam decoding, GPU quantization, and short silence
    windows favor response latency. VAD still handles interruption immediately,
    before Whisper has finished transcribing the complete utterance.
    """

    def __init__(
        self,
        interruption_controller: InterruptionController,
        *,
        model: str = "tiny.en",
        device: str = "cuda",
        compute_type: str = "int8_float16",
    ) -> None:
        self._interruption_controller = interruption_controller
        self._model = model
        self._device = device
        self._compute_type = compute_type
        self._recorder = None

    def start(self) -> None:
        """Load RealtimeSTT and begin monitoring microphone voice activity."""

        if AudioToTextRecorder is None:
            raise RuntimeError(
                'RealtimeSTT is not installed. Run: pip install "RealtimeSTT[faster-whisper]"'
            )

        self._recorder = AudioToTextRecorder(
            model=self._model,
            language="en",
            device=self._device,
            compute_type=self._compute_type,
            beam_size=1,
            beam_size_realtime=1,
            enable_realtime_transcription=False,
            faster_whisper_vad_filter=False,
            post_speech_silence_duration=0.25,
            min_length_of_recording=0.10,
            min_gap_between_recordings=0.05,
            pre_recording_buffer_duration=0.20,
            webrtc_sensitivity=3,
            silero_sensitivity=0.50,
            ensure_sentence_ends_with_period=False,
            spinner=False,
            level=logging.ERROR,
            on_vad_start=self._interruption_controller.request_interrupt,
        )

    async def listen_forever(
        self,
        prompt_queue: asyncio.Queue[PromptRequest],
        stop_event: asyncio.Event,
    ) -> None:
        """Wait for completed voice turns without blocking the asyncio loop."""

        if self._recorder is None:
            raise RuntimeError("SpeechToTextListener.start() must be called first.")

        while not stop_event.is_set():
            transcript = await asyncio.to_thread(self._recorder.text)
            cleaned_transcript = remove_filler_words(transcript)
            if not cleaned_transcript:
                continue

            # Echo the recognized request so the user can verify the STT result.
            terminal_print(f"You (voice): {cleaned_transcript}")
            request = PromptRequest(cleaned_transcript, "voice")
            await prompt_queue.put(request)
            await request.accepted.wait()

    async def shutdown(self) -> None:
        """Release RealtimeSTT's microphone and transcription worker processes."""

        if self._recorder is None:
            return
        recorder, self._recorder = self._recorder, None
        await asyncio.to_thread(recorder.shutdown)


async def collect_typed_input(
    prompt_queue: asyncio.Queue[PromptRequest],
    interruption_controller: InterruptionController,
    stop_event: asyncio.Event,
) -> None:
    """Continuously accept text so it can interrupt generation at any time."""

    while not stop_event.is_set():
        text = (await asyncio.to_thread(_TERMINAL.read_line)).strip()
        if not text:
            # An empty Enter is ignored, so the current response continues.
            continue

        interruption_controller.request_interrupt()
        request = PromptRequest(text, "text")
        await prompt_queue.put(request)

        # The model loop prints "thinking..." before allowing the next
        # "You: " prompt to appear on the following line.
        await request.accepted.wait()


__all__ = [
    "InterruptionController",
    "PromptRequest",
    "SpeechToTextListener",
    "collect_typed_input",
    "contextualize_interruption",
    "remove_filler_words",
    "stop_terminal_input",
    "terminal_print",
]
