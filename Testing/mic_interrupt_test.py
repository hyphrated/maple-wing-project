import asyncio


async def fake_thinking():
    try:
        while True:
            print("Fern is thinking...")
            await asyncio.sleep(1)  # Pause for 1 second before treturning to print
           
    except asyncio.CancelledError:
        print("Fern stopped thinking.")
        raise


async def main():
    # Start "Fern" in the background
    thinking_task = asyncio.create_task(fake_thinking())

    # Python waits for you to press Enter
    await asyncio.to_thread(
        input,
        "Press ENTER to interrupt Fern...\n"
    )

    # Stop the background task
    thinking_task.cancel()

    try:
        await thinking_task
    except asyncio.CancelledError:
        pass


asyncio.run(main())