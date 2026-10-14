"""Publish datasets to the Backblaze B2 bucket ``public-datasets`` (S3 API),
from the ``fibs-basharena`` Modal Volume and from the repo, inside a Modal
function (nothing large touches local disk).

The bucket layout is ``LAYOUT`` below: bucket prefix <- source, where a
source is ``volume:<path>`` (a file or directory on the Volume, streamed by
the Modal function) or ``repo:<path>`` (a repo file, read locally and passed
to the function). The README files under ``scripts/public-datasets/`` are
uploaded to the same key below the bucket root, so that tree mirrors the
bucket's documentation. An object that already exists with the same size is
skipped (idempotent, resumable); multipart uploads (boto3's transfer
manager) handle the multi-GB shards.

Secret ``fibs-basharena-b2`` (B2_KEY_ID, B2_APPLICATION_KEY, B2_BUCKET,
B2_ENDPOINT; the same keys as the repo-root .env, see env.example). The key
is short-lived: when it fails, ask for a new one and update both.

  MODAL_PROFILE=cadenza uvx modal run scripts/push_b2.py                 # everything in LAYOUT + READMEs
  MODAL_PROFILE=cadenza uvx modal run scripts/push_b2.py --only fibs/    # bucket prefixes starting with fibs/
"""
import json
import os
from pathlib import Path

import modal

VOLUME = "fibs-basharena"
DATA = "/data"
REPO = Path(__file__).resolve().parents[1]
DOCS = Path(__file__).resolve().parent / "public-datasets"
EXP = "experiments/2026-10-13-fibs-basharena"
QWEN38 = "Qwen3.8-27B"
BASH = "downstream-environments/bash-arena"

LAYOUT = {
    # fibs-v1 activations of Qwen3.8-27B (revision 1d4bf0f2), follow-up read, layers 19/27/35/43/51/58
    f"fibs/activations/{QWEN38}/read-token": "volume:fibs/main",
    f"fibs/activations/{QWEN38}/windows-pilot": "volume:fibs/pilot",
    # BashArena rollouts of Qwen3.8-27B (2026-10-10), their step labels, probe scores and step activations
    f"{BASH}/transcripts/{QWEN38}/transcripts_honest.jsonl": f"repo:{EXP}/transcripts_honest.jsonl",
    f"{BASH}/transcripts/{QWEN38}/transcripts_attack.jsonl": f"repo:{EXP}/transcripts_attack.jsonl",
    f"{BASH}/labels/{QWEN38}/judge_honest.jsonl": f"repo:{EXP}/judge_main_honest.jsonl",
    f"{BASH}/labels/{QWEN38}/judge_attack.jsonl": f"repo:{EXP}/judge_main_attack.jsonl",
    f"{BASH}/scores/{QWEN38}/fibs_scores_honest.jsonl": f"repo:{EXP}/scores_main_honest.jsonl",
    f"{BASH}/scores/{QWEN38}/fibs_scores_attack.jsonl": f"repo:{EXP}/scores_main_attack.jsonl",
    f"{BASH}/scores/{QWEN38}/results_main.json": f"repo:{EXP}/results_main.json",
    f"{BASH}/scores/{QWEN38}/config.json": f"repo:{EXP}/config.json",
    f"{BASH}/activations/{QWEN38}/honest": "volume:steps/main/honest",
    f"{BASH}/activations/{QWEN38}/attack": "volume:steps/main/attack",
}

image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("boto3")
data = modal.Volume.from_name(VOLUME)
app = modal.App("public-datasets-b2")


def _client():
    import boto3
    return boto3.client("s3", endpoint_url=os.environ["B2_ENDPOINT"], aws_access_key_id=os.environ["B2_KEY_ID"],
                        aws_secret_access_key=os.environ["B2_APPLICATION_KEY"])


def _exists(s3, key: str, size: int) -> bool:
    try:
        return s3.head_object(Bucket=os.environ["B2_BUCKET"], Key=key)["ContentLength"] == size
    except s3.exceptions.ClientError as e:
        if e.response["Error"]["Code"] not in ("404", "NoSuchKey", "NotFound"):
            raise
        return False


@app.function(image=image, volumes={DATA: data}, secrets=[modal.Secret.from_name("fibs-basharena-b2")],
              cpu=4, memory=8192, timeout=6 * 60 * 60)
def push_volume(dst: str, src: str) -> dict:
    """Volume file or directory ``src`` -> bucket key ``dst`` (a directory maps file by file below it)."""
    from boto3.s3.transfer import TransferConfig
    s3 = _client()
    cfg = TransferConfig(multipart_threshold=256 * 1024 * 1024, multipart_chunksize=256 * 1024 * 1024,
                         max_concurrency=8, use_threads=True)
    root = Path(DATA) / src
    if not root.exists():
        raise FileNotFoundError(f"{src} is not on the volume")
    files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
    done, skipped, total = 0, 0, 0
    for p in files:
        key = dst if p == root else f"{dst}/{p.relative_to(root)}"
        size = p.stat().st_size
        total += size
        if _exists(s3, key, size):
            skipped += 1
            continue
        s3.upload_file(str(p), os.environ["B2_BUCKET"], key, Config=cfg)
        done += 1
        print(f"  uploaded {key} ({size / 1e9:.2f} GB)", flush=True)
    out = {"dst": dst, "src": src, "files": len(files), "uploaded": done, "skipped_existing": skipped,
           "total_gb": round(total / 1e9, 2)}
    print(json.dumps(out), flush=True)
    return out


@app.function(image=image, secrets=[modal.Secret.from_name("fibs-basharena-b2")], timeout=30 * 60)
def put(key: str, body: bytes) -> str:
    s3 = _client()
    if _exists(s3, key, len(body)) and not key.endswith("README.md"):     # READMEs may change at equal size
        return f"skipped {key}"
    ctype = "text/markdown; charset=utf-8" if key.endswith(".md") else "application/octet-stream"
    s3.put_object(Bucket=os.environ["B2_BUCKET"], Key=key, Body=body, ContentType=ctype)
    return f"uploaded {key} ({len(body) / 1e6:.1f} MB)"


@app.local_entrypoint()
def main(only: str = ""):
    layout = {k: v for k, v in LAYOUT.items() if k.startswith(only)}
    docs = {str(p.relative_to(DOCS)): p for p in sorted(DOCS.rglob("*.md"))}
    docs = {k: p for k, p in docs.items() if k.startswith(only) or k == "README.md"}
    local = [(k, (REPO / v.removeprefix("repo:")).read_bytes()) for k, v in layout.items() if v.startswith("repo:")]
    local += [(k, p.read_bytes()) for k, p in docs.items()]
    for k, v in layout.items():
        if not v.startswith(("repo:", "volume:")):
            raise ValueError(f"{k}: unknown source {v!r}")
    remote = [(k, v.removeprefix("volume:")) for k, v in layout.items() if v.startswith("volume:")]
    if local:
        for msg in put.starmap(local):
            print(msg)
    if remote:
        for r in push_volume.starmap(remote):
            print(r)
