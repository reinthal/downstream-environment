"""Pulumi dynamic providers for RunPod serverless resources.

Talks to RunPod's REST API (https://rest.runpod.io/v1) directly because the
official Pulumi provider (runpodinfra 1.9.99) is broken for endpoints: its
saveEndpoint GraphQL mutation declares `$scalerValue: Int` while RunPod's
API requires Float in that position, so every endpoint create fails —
document-level validation, so it fails even with scaler_value unset.

This is our own finding (2026-10-07), not documented anywhere upstream;
reported with a curl repro in
https://github.com/runpod/pulumi-runpod-native/issues/35. Repro: replay the
provider's mutation with `$scalerValue: Int` vs `Float` — Int is rejected
(GRAPHQL_VALIDATION_FAILED), Float passes. Inline literals coerce Int->Float
(GraphQL spec 5.8.5) which is why RunPod's own doc examples still work and
the breakage is specific to variable-based clients like this provider.
Revisit runpodinfra if the issue gets fixed.

Auth: RUNPOD_API_KEY env var, read at operation time in the dynamic-provider
process (never stored in Pulumi state). KeyError if unset — intentional.

Update semantics: templates are replaced (delete-before-replace: RunPod
template names are unique); endpoints are updated in place via PATCH so the
endpoint id (and its OpenAI base URL) survives config changes.
"""

import os

import pulumi.dynamic as dyn

API = "https://rest.runpod.io/v1"


def _canon(value):
    """Undo Pulumi's protobuf number marshalling (every int arrives as float);
    RunPod's Go API rejects 100.0 where an int32 is expected."""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, dict):
        return {k: _canon(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_canon(v) for v in value]
    return value


def _request(method, path, body=None):
    import requests

    r = requests.request(
        method,
        f"{API}{path}",
        json=_canon(body),
        headers={"Authorization": f"Bearer {os.environ['RUNPOD_API_KEY']}"},
        timeout=60,
    )
    if not r.ok:
        raise RuntimeError(f"RunPod API {method} {path} -> {r.status_code}: {r.text}")
    return r.json() if r.text else None


class _TemplateProvider(dyn.ResourceProvider):
    def create(self, props):
        data = _request("POST", "/templates", props["body"])
        return dyn.CreateResult(id_=data["id"], outs={**props, "templateId": data["id"]})

    def diff(self, id, olds, news):
        # Update in place: a template attached to a serverless endpoint
        # cannot be deleted ("Template is associated with AI API ..."), so
        # replacement would deadlock; PATCH also sidesteps the unique-name
        # constraint on create.
        return dyn.DiffResult(changes=olds["body"] != news["body"])

    def update(self, id, olds, news):
        # isServerless is create-only; PATCH rejects it ("Extra input keys").
        body = {k: v for k, v in news["body"].items() if k != "isServerless"}
        _request("PATCH", f"/templates/{id}", body)
        return dyn.UpdateResult(outs={**news, "templateId": id})

    def delete(self, id, props):
        _request("DELETE", f"/templates/{id}")


class _EndpointProvider(dyn.ResourceProvider):
    def create(self, props):
        data = _request("POST", "/endpoints", props["body"])
        return dyn.CreateResult(id_=data["id"], outs={**props, "endpointId": data["id"]})

    def diff(self, id, olds, news):
        return dyn.DiffResult(changes=olds["body"] != news["body"])

    def update(self, id, olds, news):
        # computeType is create-only; PATCH rejects it ("Extra input keys").
        body = {k: v for k, v in news["body"].items() if k != "computeType"}
        _request("PATCH", f"/endpoints/{id}", body)
        return dyn.UpdateResult(outs={**news, "endpointId": id})

    def delete(self, id, props):
        _request("DELETE", f"/endpoints/{id}")


class _NetworkVolumeProvider(dyn.ResourceProvider):
    def create(self, props):
        data = _request("POST", "/networkvolumes", props["body"])
        return dyn.CreateResult(id_=data["id"], outs={**props, "volumeId": data["id"]})

    def diff(self, id, olds, news):
        # A volume cannot move between data centers: that replaces it (and
        # drops what is cached on it). name/size PATCH in place.
        return dyn.DiffResult(
            changes=olds["body"] != news["body"],
            replaces=["body"]
            if olds["body"]["dataCenterId"] != news["body"]["dataCenterId"]
            else [],
            delete_before_replace=False,
        )

    def update(self, id, olds, news):
        body = {k: v for k, v in news["body"].items() if k != "dataCenterId"}
        _request("PATCH", f"/networkvolumes/{id}", body)
        return dyn.UpdateResult(outs={**news, "volumeId": id})

    def delete(self, id, props):
        _request("DELETE", f"/networkvolumes/{id}")


class NetworkVolume(dyn.Resource):
    def __init__(self, name, body, opts=None):
        super().__init__(_NetworkVolumeProvider(), name, {"body": body, "volumeId": None}, opts)


class Template(dyn.Resource):
    def __init__(self, name, body, opts=None):
        super().__init__(_TemplateProvider(), name, {"body": body, "templateId": None}, opts)


class Endpoint(dyn.Resource):
    def __init__(self, name, body, opts=None):
        super().__init__(_EndpointProvider(), name, {"body": body, "endpointId": None}, opts)
