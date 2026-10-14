"""Pulumi dynamic providers for Modal resources (Volumes and Secrets), over
the Modal Python SDK — there is no official Pulumi provider for Modal.

Auth and workspace: the Modal token comes from ``~/.modal.toml`` (or the
``MODAL_TOKEN_ID`` / ``MODAL_TOKEN_SECRET`` env vars) in the dynamic-provider
process, and the workspace is selected by the ``MODAL_PROFILE`` env var at
operation time. Every resource names its ``profile``; the provider refuses
to act when ``MODAL_PROFILE`` differs, so a stale shell cannot create the
resources in the wrong workspace. Nothing about the token is stored in
Pulumi state.

Volumes: ``create`` is idempotent (``create_if_missing``), so a volume made
by hand with the same name is adopted rather than duplicated; ``delete``
removes the volume AND ITS DATA (Modal's ``Volume.delete``). A rename
replaces the resource.

Secrets: ``values`` maps each secret key to the environment variable its
value is read from; ``create`` reads them at operation time and never stores
a value in state; ``update`` re-creates the secret with the current values
when the inputs change (the stored inputs are only the names).
"""

import os

import pulumi.dynamic as dyn


def _check_profile(profile):
    active = os.environ.get("MODAL_PROFILE")
    if active != profile:
        raise RuntimeError(
            f"resource is declared for Modal profile {profile!r} but MODAL_PROFILE={active!r}; "
            f"run with MODAL_PROFILE={profile}"
        )


class _VolumeProvider(dyn.ResourceProvider):
    def create(self, props):
        import modal

        _check_profile(props["profile"])
        vol = modal.Volume.from_name(props["name"], create_if_missing=True)
        vol.hydrate()
        return dyn.CreateResult(id_=f"{props['profile']}/{props['name']}", outs={**props, "volumeId": vol.object_id})

    def diff(self, id_, olds, news):
        replaces = [k for k in ("name", "profile") if olds.get(k) != news.get(k)]
        return dyn.DiffResult(changes=bool(replaces), replaces=replaces, delete_before_replace=False)

    def delete(self, id_, props):
        import modal

        _check_profile(props["profile"])
        modal.Volume.objects.delete(props["name"])


class _SecretProvider(dyn.ResourceProvider):
    def _put(self, props):
        import modal

        _check_profile(props["profile"])
        env = {k: os.environ[v] for k, v in props["values"].items()}   # KeyError if unset — intentional
        modal.Secret.objects.create(props["name"], env, allow_existing=True)

    def create(self, props):
        self._put(props)
        return dyn.CreateResult(id_=f"{props['profile']}/{props['name']}", outs=dict(props))

    def diff(self, id_, olds, news):
        replaces = [k for k in ("name", "profile") if olds.get(k) != news.get(k)]
        changed = any(olds.get(k) != news.get(k) for k in ("name", "profile", "values"))
        return dyn.DiffResult(changes=changed, replaces=replaces, delete_before_replace=True)

    def update(self, id_, olds, news):
        self._put(news)
        return dyn.UpdateResult(outs=dict(news))

    def delete(self, id_, props):
        import modal

        _check_profile(props["profile"])
        modal.Secret.objects.delete(props["name"])


class Volume(dyn.Resource):
    """A Modal Volume ``name`` in the workspace of Modal profile ``profile``."""

    def __init__(self, resource_name, name, profile, opts=None):
        super().__init__(_VolumeProvider(), resource_name,
                         {"name": name, "profile": profile, "volumeId": None}, opts)


class Secret(dyn.Resource):
    """A Modal Secret ``name`` whose keys are read at ``pulumi up`` time from
    environment variables: ``values`` maps secret key -> environment variable."""

    def __init__(self, resource_name, name, profile, values, opts=None):
        super().__init__(_SecretProvider(), resource_name,
                         {"name": name, "profile": profile, "values": values}, opts)
