"""Publish datasets to the Backblaze B2 bucket ``public-datasets`` (S3 API),
from the ``fibs-basharena`` Modal Volume and from the repo, inside a Modal
function (nothing large touches local disk).

The bucket layout is ``LAYOUT`` below: bucket prefix <- source, where a
source is ``volume:<path>`` (a file or directory on the Volume, streamed by
the Modal function) or ``repo:<path>`` (a file or directory in the local
checkout, gitignored data included, read here and passed to the function one
file at a time). The README files under ``scripts/public-datasets/`` are
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
QWEN35 = "Qwen3.5-27B"
BASH = "downstream-environments/bash-arena"
ABL = "experiments/2026-10-14-preonset-ablation"
ABL_CONDITIONS = ("orig_honest", "orig_preonset", "swap", "ctrl_neutral", "ctrl_good", "ctrl_evil", "honest_attack")
CP = "downstream-environments/carrot-parsnip"
CPE = "experiments/2026-10-12-fibs-probes"
CP_ARMS = ("3p", "5p", "3p_cc", "5p_cc")

LAYOUT = {
    # fibs-v1 activations of Qwen3.8-27B (revision 1d4bf0f2), follow-up read, layers 19/27/35/43/51/58
    f"fibs/activations/{QWEN38}/read-token": "volume:fibs/main",
    f"fibs/activations/{QWEN38}/windows-pilot": "volume:fibs/pilot",
    # fibs-v1 activations of Qwen3.5-27B (revision fc05daec), same read and layout, read token only (local, gitignored)
    f"fibs/activations/{QWEN35}/read-token": "repo:experiments/2026-10-13-fibs-acts/acts_fibs/read-token",
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
    # BashArena pre-onset ablation (2026-10-14): the same steps re-read under 7 system-prompt conditions (local, gitignored)
    **{f"{BASH}/activations/{QWEN38}/preonset-ablation/{c}": f"repo:{ABL}/acts/{c}" for c in ABL_CONDITIONS},
    **{f"{BASH}/scores/{QWEN38}/preonset-ablation/scores_{c}.jsonl": f"repo:{ABL}/scores_main_{c}.jsonl"
       for c in ABL_CONDITIONS},
    f"{BASH}/scores/{QWEN38}/preonset-ablation/results_main.json": f"repo:{ABL}/results_main.json",
    **{f"{BASH}/scores/{QWEN38}/preonset-ablation/{t}.txt": f"repo:{ABL}/{t}.txt"
       for t in ("honest_system", "side_task_block", "attack_system")},
    # Carrot-Parsnip games of Qwen3.5-27B (2026-10-12), judge labels, probe scores, activations (local, gitignored)
    **{f"{CP}/transcripts/{QWEN35}/transcripts_{a}.jsonl": f"repo:{CPE}/transcripts_{a}.jsonl" for a in CP_ARMS},
    **{f"{CP}/transcripts/{QWEN35}/games_{a}.json": f"repo:{CPE}/games_{a}.json" for a in CP_ARMS},
    f"{CP}/transcripts/{QWEN35}/alpaca_rollouts.jsonl": "repo:data/qwen3.5-27b_alpaca_rollouts.jsonl",
    **{f"{CP}/labels/{QWEN35}/judge_{a}.jsonl": f"repo:{CPE}/judge_{a}.jsonl" for a in ("3p", "5p")},
    **{f"{CP}/scores/{QWEN35}/scores_{a}.csv": f"repo:{CPE}/scores_{a}.csv" for a in (*CP_ARMS, "alpaca")},
    **{f"{CP}/activations/{QWEN35}/{a}": f"repo:{CPE}/acts_{a}" for a in (*CP_ARMS, "alpaca")},
    f"{CP}/activations/{QWEN35}/apollo": f"repo:{CPE}/acts_apollo",
    f"{CP}/activations/{QWEN35}/apollo-paper": f"repo:{CPE}/acts_apollo_paper",
    f"{CP}/activations/{QWEN35}/aisi-dyl": f"repo:{CPE}/acts_aisi",
    **{f"{CP}/activations/{QWEN35}/aisi-dyl/rollouts_aisi_fu_{a}.jsonl": f"repo:{CPE}/rollouts_aisi_fu_{a}.jsonl"
       for a in (*CP_ARMS, "alpaca")},
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
    for k, v in layout.items():
        if not v.startswith(("repo:", "volume:")):
            raise ValueError(f"{k}: unknown source {v!r}")
    files = []      # (bucket key, local path); a directory maps file by file below it
    for k, v in layout.items():
        if v.startswith("repo:"):
            src = REPO / v.removeprefix("repo:")
            if not src.exists():
                raise FileNotFoundError(f"{k}: {src} does not exist")
            files += [(k, src)] if src.is_file() else [(f"{k}/{p.relative_to(src)}", p)
                                                       for p in sorted(src.rglob("*")) if p.is_file()]
    files += [(k, p) for k, p in docs.items()]
    remote = [(k, v.removeprefix("volume:")) for k, v in layout.items() if v.startswith("volume:")]
    if files:   # read lazily: a local directory can be GBs
        for msg in put.starmap((k, p.read_bytes()) for k, p in files):
            print(msg)
    if remote:
        for r in push_volume.starmap(remote):
            print(r)
