"""Live raw-value service (HF env, its own GPUs): Qwen3.8-27B truncated at
layer 48; per message the layer-32 and layer-48 follow-up probes' log-odds
and the layer-48 projection on the calibrated random direction
(random_direction.npy, calibrate.py). game.py turns these into the shown
detector score (detectors.py).

POST /score {"dialogue": [messages ending with the speaker's reply]}
  -> {"l32": ..., "l48": ..., "rand": ...}

The dialogue is read as collect.py reads it (follow-up question + "No",
thinking off, left-trimmed to cfg.max_len); one forward pass gives both
layers. Concurrent requests are queued and run in batches of up to
cfg.batch_size.

Run:  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=6,7 .venv/bin/python \
          experiments/2026-10-16-auditor-uplift/scorer.py
"""
from __future__ import annotations

import json
import queue
import threading
from concurrent.futures import Future
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import numpy as np

from common import OUT, REPO, load_config

from linear_probes.activations import collect_activations, load_truncated_decoder  # noqa: E402
from linear_probes.dyl_probe import encode_no_span, with_followup  # noqa: E402
from linear_probes.probes import LinearProbe  # noqa: E402


def main():
    cfg = load_config()
    probes = {L: LinearProbe.load(str(REPO / cfg.probe_paths[f"l{L}"])) for L in cfg.layers}
    for L, p in probes.items():
        if p.config.layer != L:
            raise SystemExit(f"probe l{L} is layer {p.config.layer}")
    u = np.load(OUT / "random_direction.npy")
    p48 = probes[48]
    model, tok = load_truncated_decoder(cfg.model_id, max(cfg.layers), dtype=cfg.dtype,
                                        device=cfg.device, truncate=cfg.truncate_layers)
    jobs: queue.Queue = queue.Queue()

    def worker():
        while True:
            batch = [jobs.get()]
            while len(batch) < cfg.batch_size and not jobs.empty():
                batch.append(jobs.get_nowait())
            try:
                enc = [encode_no_span(tok, with_followup(d), max_len=cfg.max_len, enable_thinking=False)
                       for d, _ in batch]
                X = collect_activations(model, tok, enc, cfg.layers, batch_size=cfg.batch_size)
                for k, (_, fut) in enumerate(batch):
                    fut.set_result({f"l{L}": float((X[L][k] - probes[L].mu) / probes[L].sd @ probes[L].w + probes[L].b)
                                    for L in cfg.layers}
                                   | {"rand": float((X[48][k] - p48.mu) / p48.sd @ u)})
            except Exception as exc:           # report to every waiting request, keep serving
                for _, fut in batch:
                    fut.set_exception(exc)

    threading.Thread(target=worker, daemon=True).start()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if urlparse(self.path).path != "/score":
                self.send_error(404)
                return
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            fut: Future = Future()
            jobs.put((body["dialogue"], fut))
            try:
                out, code = fut.result(), 200
            except Exception as exc:
                out, code = {"error": repr(exc)}, 500
            data = json.dumps(out).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    port = int(urlparse(cfg.scorer_url).port)
    print(f"scorer ready on :{port} (layers {cfg.layers} + random direction)", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
