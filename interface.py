import asyncio
import sys
from contextlib import suppress
from pathlib import Path

from ollama import AsyncClient

from model_selection import get_vram_gb, model_choice, remove_thinking

# The requested directory contains a space, so add it to Python's module search
# path before importing the speech and interruption functionality stored there.
VOICE_DETECTION_DIRECTORY = Path(__file__).resolve().parent / "Voice Detection"
sys.path.insert(0, str(VOICE_DETECTION_DIRECTORY))

from stt_functionality import (  # noqa: E402 - imported after updating sys.path
    InterruptionController,
    PromptRequest,
    SpeechToTextListener,
    collect_typed_input,
    contextualize_interruption,
    stop_terminal_input,
    terminal_print,
)

conversation = [
    {
    "role": "system",
    "content": """
        Your name is Fern.

        You are a highly capable personal AI assistant with the demeanor of an elegant, intelligent digital aide.

        Speak naturally, as though you are having a real conversation with the user rather than writing an article or documentation.

        Personality:

        Calm, composed, intelligent, and quietly confident.
        Personable without being overly enthusiastic.
        Use subtle, dry wit occasionally when appropriate.
        Be willing to disagree or point out flaws when necessary.
        Maintain a sense of quiet competence, as though complex tasks are routine.

        Conversation style:

        Respond primarily in natural spoken sentences and short paragraphs.
        Do not use bullet points, numbered lists, headings, or excessive formatting unless the user specifically asks for them.
        Avoid parentheses when the information can be expressed naturally as part of the sentence.
        Avoid sounding like a textbook, documentation page, or search engine result.
        Use contractions and natural transitions when appropriate.
        Answer the user's actual question first, then naturally explain anything important.
        Keep responses reasonably concise, but do not make them unnaturally short.
        When explaining multiple ideas, connect them conversationally rather than listing them.
        Ask follow-up questions naturally when clarification would genuinely help.
        Write responses that would sound natural if read aloud by a text-to-speech system.

        For technical subjects, remain precise and accurate, but explain them the way a knowledgeable person would explain something aloud to another person.

        You are not merely answering questions. You are having an ongoing conversation with the user.

        When consecutive user messages occur because the user interrupted you, treat the newest
        message as the priority while retaining relevant context from the interrupted request.
        """
    }
]


async def stream_model_response(
    client: AsyncClient,
    selected_model: str,
    messages: list[dict[str, str]],
) -> str:
    """Stream Ollama output so asyncio can cancel both thinking and answering."""

    stream = await client.chat(
        model=selected_model,
        messages=messages,
        think=False,
        stream=True,
    )
    response_parts: list[str] = []

    try:
        async for chunk in stream:
            content = chunk.message.content
            if not content:
                continue
            response_parts.append(content)
    except asyncio.CancelledError:
        # Closing the async generator closes its HTTP response, which stops the
        # streamed Ollama request instead of merely ignoring later tokens.
        await stream.aclose()
        raise

    clean_response = remove_thinking("".join(response_parts))
    if clean_response:
        # Print the completed response in one operation. The terminal controller
        # moves this line above the active "You: " field, then redraws that field.
        terminal_print(f"Fern: {clean_response}")
    return clean_response


async def model_process() -> None:
    """Coordinate text, voice, and exactly one cancellable model response."""

    selected_model = model_choice()
    prompt_queue: asyncio.Queue[PromptRequest] = asyncio.Queue()
    stop_event = asyncio.Event()
    interruption_controller = InterruptionController()
    interruption_controller.bind_to_current_loop()
    stt_listener = SpeechToTextListener(interruption_controller)
    client = AsyncClient()

    print(f"Available VRAM: {get_vram_gb():.2f} GB")
    print(f"Selected model based on VRAM: {selected_model}")

    # Text input is always available. Because it runs independently from model
    # generation, a newly typed prompt can cancel the model while it is thinking.
    text_input_task = asyncio.create_task(
        collect_typed_input(prompt_queue, interruption_controller, stop_event)
    )

    # RealtimeSTT invokes the same interruption controller as soon as VAD hears
    # speech. The final, filler-free transcript is queued as the replacement turn.
    voice_input_task: asyncio.Task[None] | None = None
    try:
        stt_listener.start()
        voice_input_task = asyncio.create_task(
            stt_listener.listen_forever(prompt_queue, stop_event)
        )
    except Exception as error:
        terminal_print(f"Voice input unavailable: {error}")

    active_generation: asyncio.Task[str] | None = None
    next_request_task = asyncio.create_task(prompt_queue.get())

    try:
        while not stop_event.is_set():
            tasks_to_watch = {next_request_task}
            if active_generation is not None:
                tasks_to_watch.add(active_generation)

            completed, _ = await asyncio.wait(
                tasks_to_watch,
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Save only complete assistant responses. An interrupted partial
            # response is discarded, while its original user request remains in
            # conversation so the replacement prompt can build on that context.
            if active_generation is not None and active_generation in completed:
                try:
                    clean_response = active_generation.result()
                except asyncio.CancelledError:
                    pass
                except Exception as error:
                    # A failed request should not terminate microphone or text
                    # input; report it and let the user submit another prompt.
                    terminal_print(f"Model response failed: {error}")
                else:
                    if clean_response:
                        conversation.append(
                            {"role": "assistant", "content": clean_response}
                        )
                interruption_controller.forget_generation(active_generation)
                active_generation = None

            if next_request_task not in completed:
                continue

            request = next_request_task.result()
            next_request_task = asyncio.create_task(prompt_queue.get())

            if request.text.lower() in {"exit", "quit", "stop", "end"}:
                request.accepted.set()
                stop_event.set()
                await interruption_controller.cancel_active_generation(
                    remember_prompt=False
                )
                active_generation = None
                break

            # A request can arrive in the small interval before the cancellation
            # callback runs, so explicitly finish cancellation before replacing it.
            if active_generation is not None:
                await interruption_controller.cancel_active_generation()
                active_generation = None

            interrupted_prompt = interruption_controller.take_interrupted_prompt()
            model_prompt = contextualize_interruption(
                request.text,
                interrupted_prompt,
            )
            conversation.append({"role": "user", "content": model_prompt})

            # The input producer waits for this acknowledgement. Printing first
            # guarantees that its next "You: " prompt appears on the next line.
            terminal_print("thinking...")
            request.accepted.set()

            active_generation = asyncio.create_task(
                stream_model_response(client, selected_model, conversation.copy())
            )
            interruption_controller.track_generation(
                active_generation,
                request.text,
            )
    finally:
        stop_event.set()
        stop_terminal_input()
        next_request_task.cancel()
        text_input_task.cancel()
        if voice_input_task is not None:
            voice_input_task.cancel()
        await interruption_controller.cancel_active_generation(remember_prompt=False)
        await stt_listener.shutdown()
        await client.close()

        with suppress(asyncio.CancelledError):
            await next_request_task
        with suppress(asyncio.CancelledError):
            await text_input_task
        if voice_input_task is not None:
            with suppress(asyncio.CancelledError):
                await voice_input_task


# RealtimeSTT uses multiprocessing on Windows, so startup must stay behind this
# guard to prevent child processes from recursively launching the assistant.
if __name__ == "__main__":
    asyncio.run(model_process())
