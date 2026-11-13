#!/usr/bin/env python3
"""Check an HF token, model approvals, and Databricks secret metadata without GPUs."""

import argparse
import getpass
import http.client
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
import warnings


class CheckFailed(Exception):
    """A safe, actionable failure message with no credential values."""


def check_secret(profile, secret):
    scope, key = secret.split("/")
    try:
        result = subprocess.run(
            [
                "databricks",
                "secrets",
                "list-secrets",
                scope,
                "--profile",
                profile,
                "--output",
                "json",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        raise CheckFailed(
            "Secret check unavailable. Install/authenticate the Databricks CLI, "
            "check connectivity, and retry with the intended --profile."
        ) from None
    if result.returncode:
        raise CheckFailed(
            "Cannot list secret metadata. Check the selected workspace, scope existence, "
            "and your scope READ permission. Run databricks auth describe with this profile."
        )
    try:
        entries = json.loads(result.stdout)
        if not isinstance(entries, list):
            raise ValueError
        present = any(entry["key"] == key for entry in entries)
    except (ValueError, KeyError, TypeError):
        raise CheckFailed(
            "Secret check unavailable: unexpected CLI metadata response."
        ) from None
    if not present:
        raise CheckFailed(
            "HF secret key is missing in this workspace. Store the token with "
            "databricks secrets put-secret SCOPE KEY --profile PROFILE "
            "using the hidden interactive prompt, then retry."
        )


class NoRedirects(urllib.request.HTTPRedirectHandler):
    """Keep credentials on the fixed HF API endpoint and reject login redirects."""

    def redirect_request(self, request, response, code, message, headers, new_url):
        return None


def check_hf(path, token, model=None):
    request = urllib.request.Request(
        "https://huggingface.co" + path,
        headers={
            "Authorization": "Bearer " + token,
            "User-Agent": "air-cookbook-preflight",
        },
    )
    try:
        opener = urllib.request.build_opener(NoRedirects())
        with opener.open(request, timeout=20) as response:
            if model is None:
                identity = json.load(response)
                if not isinstance(identity, dict) or not any(
                    isinstance(identity.get(key), str) and identity[key]
                    for key in ("name", "id")
                ):
                    raise ValueError("Unexpected identity response")
    except urllib.error.HTTPError as error:
        if error.code == 401 and model is None:
            message = "HF token is invalid or expired. Create a read token and update the workspace secret."
        elif error.code in (401, 403) and model is not None:
            message = (
                f"Model access denied: https://huggingface.co/{model}. "
                "Request/accept model access with the token owner's HF account, wait for approval, "
                "and grant the token access to this repository (including gated repositories)."
            )
        elif error.code == 404 and model is not None:
            message = (
                f"Cannot establish access to https://huggingface.co/{model}. "
                "Check the model ID, private-repository permissions, and HF API availability."
            )
        else:
            message = "HF access check unavailable. Check service status/connectivity and retry before submission."
        raise CheckFailed(message) from None
    except (OSError, ValueError, http.client.HTTPException):
        raise CheckFailed(
            "HF access check unavailable. Check connectivity and retry before submission."
        ) from None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile", required=True, help="Databricks workspace CLI profile"
    )
    parser.add_argument(
        "--secret",
        default="my_scope/hf_token",
        help="HF_TOKEN scope/key in workload.yaml",
    )
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="HF owner/model ID; repeat for dependencies",
    )
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[\w.@-]+/[\w.-]+", args.secret, flags=re.ASCII):
        parser.error("--secret must be a scope/key reference")
    for model in args.model:
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", model, flags=re.ASCII):
            parser.error("--model must be an owner/model ID")
    try:
        token = os.environ.get("HF_TOKEN", "")
        if not token and sys.stdin.isatty():
            with warnings.catch_warnings():
                warnings.simplefilter("error", getpass.GetPassWarning)
                try:
                    token = getpass.getpass(
                        "HF token (same token stored in the workspace secret): "
                    )
                except getpass.GetPassWarning:
                    raise CheckFailed(
                        "A hidden token prompt is unavailable. Provide HF_TOKEN through a secret environment."
                    ) from None
        if (
            not token
            or any(character.isspace() for character in token)
            or not token.isascii()
        ):
            raise CheckFailed(
                "HF token is missing or malformed. Run interactively for a hidden prompt, "
                "or provide HF_TOKEN through your agent's secret environment."
            )
        check_hf("/api/whoami-v2", token)
        for model in dict.fromkeys(args.model):
            check_hf(f"/api/models/{model}/auth-check", token, model)
            print(f"Model access confirmed: https://huggingface.co/{model}")
        check_secret(args.profile, args.secret)
    except (CheckFailed, EOFError, KeyboardInterrupt) as error:
        print(
            (
                str(error)
                if isinstance(error, CheckFailed)
                else "Check interrupted; access was not established."
            ),
            file=sys.stderr,
        )
        return 1
    print(
        "Checks passed for the supplied token and CLI identity. Secret key metadata exists; "
        "the stored value and workload run-as permissions were not verified."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
