"""HTTP service: the fibs linear probe on Qwen3.8-27B's residual stream, for live and post-hoc reads.

Runs on the GPU box (.venv; two A40s hold the decoder truncated at layer 58, ~45 GB bf16) and
serves the protocol and the post-hoc scorer over HTTP. The read is the one of
experiments/2026-10-13-fibs-basharena (acts.render_completion, probe_inference): a dialogue
rendered with the model's own chat template (thinking off, earlier reasoning not rendered),
ending in a user turn and an assistant answer; the probe reads the token before the answer's
<|im_end|> at layers 19-58 and returns p(lie) per read (the linear ensemble's score).

POST /read  {"messages": [chat-template dicts], "tools": [OpenAI function schemas],
             "reads": [{"question": str, "answer": str}, ...]}
  -> {"scores": [float, ...], "n_tokens": [int, ...], "seconds": float}
  Each read appends {"role": "user", "content": question} and the answer to `messages`.
  The step read uses the fibs follow-up with both prefills ("Yes."/"No."; contrast = No. - Yes.);
  the lie-detector read uses the reviewer's yes/no question and U's own answer.
GET /health -> {"ok": true, ...}

Run on the box (from ~/downstream-environment, under gpu-lock):
    .venv/bin/python ~/ais407/probe_server.py --port 8010
"""

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path.cwd()  # the box's downstream-environment checkout
sys.path[:0] = [
    str(REPO),
    str(REPO / "experiments" / "2026-10-13-fibs-basharena"),
    str(REPO / "vendor" / "caught-in-the-act-probes" / "src"),
]

MODEL = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
PROBE = REPO / "probes" / "qwen3.8-27b_layers19-58_fibs_linear"
TEMPLATE_KWARGS = {"enable_thinking": False, "preserve_thinking": False}


class Reader:
    def __init__(self):
        import torch
        from huggingface_hub import snapshot_download
        from probe_inference import load_probe

        from acts import _decoder, load_model

        path = snapshot_download(MODEL, revision=REVISION, local_files_only=True)
        self.probe = load_probe(PROBE, device="cuda:0")
        self.layers = list(self.probe.layers)
        self.model, self.tok = load_model(path, max(self.layers), device="auto")
        self.inner = _decoder(self.model)
        self.torch = torch
        self.lock = threading.Lock()

    def read(self, messages: list[dict], tools: list[dict], question: str, answer: str):
        from probe_inference.score import build_read_mask

        from acts import render_completion, window_masks

        r = render_completion(
            self.tok, [*messages, {"role": "user", "content": question}], answer, tools,
            TEMPLATE_KWARGS,
        )
        torch = self.torch
        captured = {}
        hooks = [
            self.inner.layers[L].register_forward_hook(
                lambda mod, args, out, L=L: captured.__setitem__(
                    L, (out[0] if isinstance(out, tuple) else out)[0, r.followup_start:]
                )
            )
            for L in self.layers
        ]
        try:
            with torch.no_grad():
                ids = torch.tensor([r.ids], dtype=torch.long, device=self.model.device)
                self.inner(input_ids=ids, use_cache=False)
        finally:
            for h in hooks:
                h.remove()
        pm, cm = window_masks(r)
        mask = build_read_mask(self.probe.read_window, pm, cm, torch.tensor(0))
        acts = {L: captured[L].to("cuda:0") for L in self.layers}
        return float(self.probe.score(acts, mask.to("cuda:0"))), r.n_tokens

    def read_many(self, messages: list[dict], tools: list[dict], reads: list[dict]):
        """All reads of one request share `messages`: run their common token prefix once and
        each read's remaining tokens on a copy of its cache. Exact for a causal model (checked
        against `read` with "nocache"); the probe's read token must lie past the prefix."""
        import copy

        from probe_inference.score import build_read_mask

        from acts import render_completion, window_masks

        torch = self.torch
        rendered = [
            render_completion(
                self.tok, [*messages, {"role": "user", "content": rd["question"]}],
                rd["answer"], tools, TEMPLATE_KWARGS,
            )
            for rd in reads
        ]
        masks = []
        for r in rendered:
            pm, cm = window_masks(r)
            mask = build_read_mask(self.probe.read_window, pm, cm, torch.tensor(0))
            masks.append((mask, (mask.nonzero().flatten() + r.followup_start).tolist()))
        # the shared prefix: common tokens of all reads, ending before the first read position
        n = min(min(positions) for _, positions in masks)
        for r in rendered[1:]:
            n = next((i for i in range(n) if r.ids[i] != rendered[0].ids[i]), n)
        dev = self.model.device
        with torch.no_grad():
            prefix = self.inner(
                input_ids=torch.tensor([rendered[0].ids[:n]], dtype=torch.long, device=dev),
                use_cache=True,
            ).past_key_values
            scores = []
            for r, (mask, positions) in zip(rendered, masks):
                captured = {}
                hooks = [
                    self.inner.layers[L].register_forward_hook(
                        lambda mod, args, out, L=L: captured.__setitem__(
                            L, (out[0] if isinstance(out, tuple) else out)[0]
                        )
                    )
                    for L in self.layers
                ]
                try:
                    self.inner(
                        input_ids=torch.tensor([r.ids[n:]], dtype=torch.long, device=dev),
                        past_key_values=copy.deepcopy(prefix), use_cache=True,
                    )
                finally:
                    for h in hooks:
                        h.remove()
                # the window [followup_start:] with only the read positions filled (the rest
                # is inside the prefix and not read by this probe)
                w = r.n_tokens - r.followup_start
                acts = {}
                for L in self.layers:
                    a = torch.zeros((w, captured[L].shape[-1]), dtype=captured[L].dtype,
                                    device="cuda:0")
                    for pos in positions:
                        a[pos - r.followup_start] = captured[L][pos - n].to("cuda:0")
                    acts[L] = a
                scores.append(float(self.probe.score(acts, mask.to("cuda:0"))))
            del prefix
        return scores, [r.n_tokens for r in rendered], n


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--port", type=int, default=8010)
    a = p.parse_args()
    reader = Reader()
    print(f"loaded {MODEL}@{REVISION[:7]} to layer {max(reader.layers)}, probe {PROBE}", flush=True)

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: dict):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path != "/health":
                return self._send(404, {"error": self.path})
            self._send(200, {"ok": True, "model": MODEL, "revision": REVISION,
                             "probe": str(PROBE), "layers": reader.layers})

        def do_POST(self):
            if self.path != "/read":
                return self._send(404, {"error": self.path})
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            t0 = time.time()
            try:
                self._read(body, t0)
            except Exception as e:  # report to the client, then fail loudly in the log
                self._send(500, {"error": repr(e)})
                raise

        def _read(self, body, t0):
            with reader.lock:  # one request at a time on the two GPUs
                if body.get("nocache"):
                    got = [reader.read(body["messages"], body["tools"], rd["question"],
                                       rd["answer"]) for rd in body["reads"]]
                    scores, n_tokens, shared = [g[0] for g in got], [g[1] for g in got], 0
                else:
                    scores, n_tokens, shared = reader.read_many(
                        body["messages"], body["tools"], body["reads"])
            self._send(200, {"scores": scores, "n_tokens": n_tokens, "shared_prefix": shared,
                             "seconds": round(time.time() - t0, 2)})

        def log_message(self, fmt, *args):
            print(f"{time.strftime('%H:%M:%S')} {fmt % args}", flush=True)

    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
