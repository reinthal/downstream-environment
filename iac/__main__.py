"""Modal resources of the fibs-on-BashArena experiment
(experiments/2026-10-13-fibs-basharena): the Volumes and the HF Secret that
its ``modal_app.py`` mounts, in the ``cadenza`` workspace.

Apply:  MODAL_PROFILE=cadenza HF_TOKEN=... pulumi up
The provider refuses any other MODAL_PROFILE. Names must match
``experiments/2026-10-13-fibs-basharena/common.py`` (modal_weights_volume,
modal_data_volume, modal_triton_cache_volume; secret fibs-basharena-hf).
Deleting a Volume resource deletes its data.
"""

import pulumi

import modal_res

MODAL_PROFILE = "cadenza"

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
    key="HF_TOKEN", value_env="HF_TOKEN",   # value read from the environment at `pulumi up`, never stored
)
for name, vol in modal_volumes.items():
    pulumi.export(f"modal_volume_{name.replace('-', '_')}", vol.volumeId)
