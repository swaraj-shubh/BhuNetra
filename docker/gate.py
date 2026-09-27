"""Offline proof mode only: relay 0.0.0.0:8000 -> api:8000. Docker internal networks drop published ports, so this
container sits on both networks and forwards the one web port; the api and worker keep no route out (DESIGN.md §2.8).
Stdlib asyncio, run from the api image: nothing extra to stage."""

import asyncio

UPSTREAM = ("api", 8000)


async def pipe(r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
    try:
        while data := await r.read(65536):
            w.write(data)
            await w.drain()
    finally:
        w.close()


async def handle(cr: asyncio.StreamReader, cw: asyncio.StreamWriter) -> None:
    try:
        ur, uw = await asyncio.open_connection(*UPSTREAM)
    except OSError:
        cw.close()
        return
    await asyncio.gather(pipe(cr, uw), pipe(ur, cw), return_exceptions=True)


async def main() -> None:
    server = await asyncio.start_server(handle, "0.0.0.0", 8000)  # reachable only via the loopback port map
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
