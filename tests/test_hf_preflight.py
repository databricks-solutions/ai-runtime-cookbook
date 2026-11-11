"""Preflight failure modes use no credentials, network, or Databricks workspace."""

import contextlib
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import unittest
from unittest import mock
import urllib.error

SPEC = importlib.util.spec_from_file_location(
    "hf_preflight", Path(__file__).parents[1] / "scripts" / "hf_preflight.py"
)
preflight = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preflight)


class PreflightTest(unittest.TestCase):
    token = "hf_TEST_TOKEN_NEVER_PRINT"
    args = ["--profile", "test-profile", "--model", "owner/model"]

    def setUp(self):
        self.environment = mock.patch.dict(os.environ, {"HF_TOKEN": self.token})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.transport = mock.patch.object(preflight.urllib.request, "build_opener")
        self.opener = self.transport.start().return_value
        self.addCleanup(self.transport.stop)
        self.opener.open.side_effect = lambda *a, **kw: io.BytesIO(
            b'{"name":"test-user"}'
        )
        self.cli = mock.patch.object(preflight.subprocess, "run")
        self.run = self.cli.start()
        self.addCleanup(self.cli.stop)
        self.run.return_value = subprocess.CompletedProcess(
            [], 0, '[{"key":"hf_token"}]', ""
        )

    def invoke(self, args=None):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = preflight.main(self.args if args is None else args)
        output = stdout.getvalue() + stderr.getvalue()
        self.assertNotIn(self.token, output)
        return code, output

    def http_error(self, status):
        # Neither exception messages nor server bodies may escape to output.
        return urllib.error.HTTPError(
            "https://huggingface.co/",
            status,
            self.token,
            {},
            io.BytesIO(self.token.encode()),
        )

    def test_missing_token_does_not_contact_services(self):
        with mock.patch.dict(os.environ, {"HF_TOKEN": ""}), mock.patch.object(
            preflight.sys.stdin, "isatty", return_value=False
        ):
            code, output = self.invoke()
        self.assertEqual(code, 1)
        self.assertIn("missing or malformed", output)
        self.opener.open.assert_not_called()
        self.run.assert_not_called()

    def test_hidden_interactive_prompt(self):
        with mock.patch.dict(os.environ, {"HF_TOKEN": ""}), mock.patch.object(
            preflight.sys.stdin, "isatty", return_value=True
        ), mock.patch.object(preflight.getpass, "getpass", return_value=self.token):
            code, _ = self.invoke()
        self.assertEqual(code, 0)

    def test_prompt_does_not_fall_back_to_echo(self):
        with mock.patch.dict(os.environ, {"HF_TOKEN": ""}), mock.patch.object(
            preflight.sys.stdin, "isatty", return_value=True
        ), mock.patch.object(
            preflight.getpass, "getpass", side_effect=preflight.getpass.GetPassWarning
        ):
            code, output = self.invoke()
        self.assertEqual(code, 1)
        self.assertIn("hidden token prompt is unavailable", output)

    def test_invalid_token(self):
        self.opener.open.side_effect = self.http_error(401)
        code, output = self.invoke()
        self.assertEqual(code, 1)
        self.assertIn("invalid or expired", output)
        self.run.assert_not_called()

    def test_model_approval_denied(self):
        for status in (401, 403):
            with self.subTest(status=status):
                self.opener.open.side_effect = [
                    io.BytesIO(b'{"name":"user"}'),
                    self.http_error(status),
                ]
                code, output = self.invoke()
                self.assertEqual(code, 1)
                self.assertIn("https://huggingface.co/owner/model", output)
                self.assertIn("wait for approval", output)
                self.assertNotIn("invalid or expired", output)

    def test_two_models_and_metadata(self):
        code, output = self.invoke(
            self.args + ["--model", "other/model", "--model", "owner/model"]
        )
        self.assertEqual(code, 0)
        requests = self.opener.open.call_args_list
        self.assertEqual(
            [call.args[0].full_url for call in requests],
            [
                "https://huggingface.co/api/whoami-v2",
                "https://huggingface.co/api/models/owner/model/auth-check",
                "https://huggingface.co/api/models/other/model/auth-check",
            ],
        )
        for call in requests:
            self.assertEqual(
                call.args[0].get_header("Authorization"), "Bearer " + self.token
            )
            self.assertEqual(call.kwargs["timeout"], 20)
        self.run.assert_called_once_with(
            [
                "databricks",
                "secrets",
                "list-secrets",
                "my_scope",
                "--profile",
                "test-profile",
                "--output",
                "json",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertIn(
            "stored value and workload run-as permissions were not verified", output
        )

    def test_missing_secret_key(self):
        self.run.return_value.stdout = '[{"key":"other-key"}]'
        code, output = self.invoke()
        self.assertEqual(code, 1)
        self.assertIn("secret key is missing", output)
        self.assertIn("put-secret", output)

    def test_metadata_permission_failure_does_not_print_cli_output(self):
        self.run.return_value = subprocess.CompletedProcess(
            [], 1, self.token, self.token
        )
        code, output = self.invoke()
        self.assertEqual(code, 1)
        self.assertIn("Cannot list secret metadata", output)
        self.assertIn("READ permission", output)

    def test_cli_unavailable_and_timeout(self):
        for failure in (
            OSError(self.token),
            subprocess.TimeoutExpired("databricks", 30, self.token, self.token),
        ):
            with self.subTest(failure=type(failure).__name__):
                self.run.side_effect = failure
                code, output = self.invoke()
                self.assertEqual(code, 1)
                self.assertIn("Secret check unavailable", output)

    def test_malformed_metadata(self):
        for payload in (self.token, "{}", "[{}]", "[null]", '"unexpected"'):
            with self.subTest(payload=payload):
                self.run.return_value.stdout = payload
                code, output = self.invoke()
                self.assertEqual(code, 1)
                self.assertIn("unexpected CLI metadata response", output)

    def test_network_and_timeout_are_unavailable_not_denied(self):
        for failure in (
            urllib.error.URLError(self.token),
            TimeoutError(self.token),
            preflight.http.client.BadStatusLine(self.token),
        ):
            with self.subTest(failure=type(failure).__name__):
                self.opener.open.side_effect = failure
                code, output = self.invoke()
                self.assertEqual(code, 1)
                self.assertIn("check unavailable", output)
                self.assertNotIn("access denied", output)

    def test_unexpected_identity_response(self):
        for payload in (
            self.token.encode(),
            b"{}",
            b"[]",
            b'{"name":null}',
            b"<html>login</html>",
            b"\xff",
        ):
            with self.subTest(payload=payload):
                self.opener.open.side_effect = lambda *a, **kw: io.BytesIO(payload)
                code, output = self.invoke()
                self.assertEqual(code, 1)
                self.assertIn("check unavailable", output)
                self.run.assert_not_called()

    def test_missing_model_and_service_failure(self):
        for status in (404, 429, 503, 302):
            with self.subTest(status=status):
                self.opener.open.side_effect = [
                    io.BytesIO(b'{"name":"user"}'),
                    self.http_error(status),
                ]
                code, output = self.invoke()
                self.assertEqual(code, 1)
                self.assertTrue(
                    "Cannot establish access" in output or "check unavailable" in output
                )

    def test_redirects_are_rejected(self):
        handler = preflight.NoRedirects()
        self.assertIsNone(
            handler.redirect_request(
                None, None, 302, "redirect", {}, "https://other.example"
            )
        )


if __name__ == "__main__":
    unittest.main()
