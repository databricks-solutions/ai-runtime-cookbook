"""Round-robin router in front of vLLM instances, for both benchmark layouts.

colocated: each /v1/completions request goes unchanged to the next instance.
disagg: the request goes to the next prefill instance with max_tokens=1 and
kv_transfer_params.do_remote_decode set. The response's kv_transfer_params are
then attached to the request sent to the next decode instance, which streams
the answer back. This is the protocol of vLLM's
tests/v1/kv_connector/nixl_integration/toy_proxy_server.py at v0.34.0.
"""

import argparse
import itertools

import aiohttp
from aiohttp import web

COMPLETIONS_PATH = "/v1/completions"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["colocated", "disagg"], required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--instances", nargs="*", default=[], help="colocated: host:port")
    parser.add_argument("--prefill", nargs="*", default=[], help="disagg: host:port")
    parser.add_argument("--decode", nargs="*", default=[], help="disagg: host:port")
    return parser.parse_args()


class Router:
    def __init__(self, args):
        self.mode = args.mode
        self.colocated = itertools.cycle(args.instances)
        self.prefill = itertools.cycle(args.prefill)
        self.decode = itertools.cycle(args.decode)
        self.requests = 0
        self.missing_kv_params = 0

    async def on_startup(self, app):
        # No client-side connection limit: a queue here would add latency the benchmark is meant to measure.
        app["session"] = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=0),
            timeout=aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=600),
        )

    async def on_cleanup(self, app):
        await app["session"].close()

    async def health(self, request):
        return web.json_response(
            {"mode": self.mode, "requests": self.requests, "missing_kv_params": self.missing_kv_params}
        )

    async def completions(self, request):
        body = await request.json()
        self.requests += 1
        if self.mode == "colocated":
            return await self.relay(request, f"http://{next(self.colocated)}{COMPLETIONS_PATH}", body)

        prefill_body = dict(body)
        prefill_body.update(
            {
                "stream": False,
                "max_tokens": 1,
                "kv_transfer_params": {
                    "do_remote_decode": True,
                    "do_remote_prefill": False,
                    "remote_engine_id": None,
                    "remote_block_ids": None,
                    "remote_host": None,
                    "remote_port": None,
                },
            }
        )
        prefill_body.pop("stream_options", None)
        session = request.app["session"]
        url = f"http://{next(self.prefill)}{COMPLETIONS_PATH}"
        async with session.post(url, json=prefill_body) as upstream:
            if upstream.status != 200:
                return web.Response(status=upstream.status, text=await upstream.text())
            prefill_out = await upstream.json()

        kv_transfer_params = prefill_out.get("kv_transfer_params")
        if not kv_transfer_params:
            # Without these the decode instance would recompute the prompt, which would hide the KV transfer.
            self.missing_kv_params += 1
            return web.Response(status=502, text="prefill response had no kv_transfer_params")

        decode_body = dict(body)
        decode_body["kv_transfer_params"] = kv_transfer_params
        return await self.relay(request, f"http://{next(self.decode)}{COMPLETIONS_PATH}", decode_body)

    async def relay(self, request, url, body):
        session = request.app["session"]
        async with session.post(url, json=body) as upstream:
            if upstream.status != 200:
                return web.Response(status=upstream.status, body=await upstream.read())
            response = web.StreamResponse(
                status=200,
                headers={"Content-Type": upstream.headers.get("Content-Type", "text/event-stream")},
            )
            await response.prepare(request)
            async for chunk in upstream.content.iter_any():
                await response.write(chunk)
            await response.write_eof()
            return response


def main():
    args = parse_args()
    if args.mode == "colocated" and not args.instances:
        raise ValueError("colocated mode needs --instances")
    if args.mode == "disagg" and not (args.prefill and args.decode):
        raise ValueError("disagg mode needs --prefill and --decode")

    router = Router(args)
    app = web.Application()
    app.on_startup.append(router.on_startup)
    app.on_cleanup.append(router.on_cleanup)
    app.router.add_post(COMPLETIONS_PATH, router.completions)
    app.router.add_get("/healthz", router.health)
    web.run_app(app, host="127.0.0.1", port=args.port, print=None, access_log=None)


if __name__ == "__main__":
    main()
