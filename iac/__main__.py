"""Modal resources of the fibs-on-BashArena experiment
(experiments/2026-10-13-fibs-basharena): the Volumes and the HF Secret that
its ``modal_app.py`` mounts, plus the B2 secret of ``scripts/push_b2.py``, one stack per workspace (``reinthal``, ``cadenza``).

Apply:  pulumi stack select <cadenza|reinthal>; MODAL_PROFILE=<same> HF_TOKEN=... pulumi up
The provider refuses a MODAL_PROFILE that differs from the stack's `modal-iac:profile`. Names must match
``experiments/2026-10-13-fibs-basharena/common.py`` (modal_weights_volume,
modal_data_volume, modal_triton_cache_volume; secret fibs-basharena-hf).
Deleting a Volume resource deletes its data.
"""

import pulumi

import modal_res

# One stack per workspace: `pulumi stack select cadenza|reinthal`; the stack config
# `modal-iac:profile` names the Modal profile, and MODAL_PROFILE must match at apply time.
MODAL_PROFILE = pulumi.Config().require("profile")

modal_volumes = {
    name: modal_res.Volume(f"modal-volume-{name}", name, MODAL_PROFILE)
    for name in (
        "qwen35-27b-weights",     # HF cache with Qwen3.5-27B and Qwen3.8-27B (modal_app.py::fetch_weights)
        "fibs-basharena",         # rendered fibs rows, activation shards, step activations, trained probes
        "fibs-basharena-triton",  # compiled Triton kernels, reused across containers
    )
}
modal_hf_secret = modal_res.Secret(
    "modal-secret-fibs-basharena-hf", "fibs-basharena-hf", MODAL_PROFILE,
    values={"HF_TOKEN": "HF_TOKEN"},   # values read from the environment at `pulumi up`, never stored
)
# Backblaze B2 (bucket public-datasets) for scripts/push_b2.py; the application key is short-lived
modal_b2_secret = modal_res.Secret(
    "modal-secret-fibs-basharena-b2", "fibs-basharena-b2", MODAL_PROFILE,
    values={k: k for k in ("B2_KEY_ID", "B2_APPLICATION_KEY", "B2_BUCKET", "B2_ENDPOINT")},
)
for name, vol in modal_volumes.items():
    pulumi.export(f"modal_volume_{name.replace('-', '_')}", vol.volumeId)
