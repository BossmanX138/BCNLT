from __future__ import annotations

import subprocess
import time
import unittest
from unittest import mock

import pytest

from bosscrafts_model.scientific_sandbox import (
    DockerScientificSandbox,
    SandboxExecutionError,
)

pytestmark = pytest.mark.requires_docker


class ScientificSandboxIntegrationTests(unittest.TestCase):
    def test_cleanup_timeout_is_reported_as_a_sandbox_failure(self) -> None:
        sandbox = DockerScientificSandbox.load_default()
        with (
            mock.patch.object(sandbox, "_create_container", return_value="container-id"),
            mock.patch.object(
                sandbox,
                "_inspect_container",
                side_effect=ValueError("fixture inspection failure"),
            ),
            mock.patch(
                "bosscrafts_model.scientific_sandbox.subprocess.run",
                side_effect=subprocess.TimeoutExpired(["docker", "rm"], 10),
            ),
        ):
            with self.assertRaisesRegex(
                SandboxExecutionError,
                "sandbox container cleanup failed: docker rm timed out",
            ):
                sandbox.execute(language="python", source="print(1)\n", stdin_text="")

    def test_cleanup_launch_failure_is_reported_as_a_sandbox_failure(self) -> None:
        sandbox = DockerScientificSandbox.load_default()
        with (
            mock.patch.object(sandbox, "_create_container", return_value="container-id"),
            mock.patch.object(
                sandbox,
                "_inspect_container",
                side_effect=ValueError("fixture inspection failure"),
            ),
            mock.patch(
                "bosscrafts_model.scientific_sandbox.subprocess.run",
                side_effect=OSError("fixture docker unavailable"),
            ),
        ):
            with self.assertRaisesRegex(
                SandboxExecutionError,
                "sandbox container cleanup failed: docker rm could not start",
            ):
                sandbox.execute(language="python", source="print(1)\n", stdin_text="")

    def test_python_execution_is_isolated_attested_and_disposable(self) -> None:
        sandbox = DockerScientificSandbox.load_default()
        result = sandbox.execute(
            language="python",
            source=(
                "import json, sys\n"
                "payload = json.load(sys.stdin)\n"
                "print(json.dumps({'sum': payload['left'] + payload['right']}, "
                "sort_keys=True, separators=(',', ':')))\n"
            ),
            stdin_text='{"left":2,"right":3}',
        )

        self.assertEqual(result.status, "passed")
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(result.stdout_text, '{"sum":5}\n')
        self.assertEqual(result.stderr_text, "")
        self.assertEqual(result.attestation.network_mode, "none")
        self.assertTrue(result.attestation.read_only_rootfs)
        self.assertEqual(result.attestation.user, "65532:65532")
        self.assertEqual(result.attestation.mount_count, 0)
        self.assertEqual(result.attestation.cap_drop, ("ALL",))
        self.assertTrue(result.attestation.no_new_privileges)
        self.assertEqual(
            result.attestation.image_id,
            "sha256:4ca37835c5cd68d5b61b66d84296f0e0b9e003de755e20472d35299af6751234",
        )
        inspected = subprocess.run(
            ["docker", "inspect", result.container_name],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(inspected.returncode, 0)

    def test_container_cannot_reach_an_external_network(self) -> None:
        result = DockerScientificSandbox.load_default().execute(
            language="python",
            source=(
                "import socket\n"
                "try:\n"
                "    connection = socket.create_connection(('1.1.1.1', 53), timeout=0.5)\n"
                "except OSError:\n"
                "    print('blocked')\n"
                "else:\n"
                "    connection.close()\n"
                "    print('connected')\n"
            ),
            stdin_text="",
        )

        self.assertEqual(result.status, "passed")
        self.assertEqual(result.stdout_text, "blocked\n")

    def test_container_cannot_write_outside_its_tmpfs_workspace(self) -> None:
        result = DockerScientificSandbox.load_default().execute(
            language="python",
            source=(
                "try:\n"
                "    open('/forbidden-hostless-write', 'w', encoding='utf-8').write('x')\n"
                "except OSError:\n"
                "    print('blocked')\n"
                "else:\n"
                "    print('writable')\n"
            ),
            stdin_text="",
        )

        self.assertEqual(result.status, "passed")
        self.assertEqual(result.stdout_text, "blocked\n")

    def test_infinite_program_is_killed_by_the_inner_deadline(self) -> None:
        started = time.monotonic()
        result = DockerScientificSandbox.load_default().execute(
            language="python",
            source="while True:\n    pass\n",
            stdin_text="",
        )
        elapsed = time.monotonic() - started

        self.assertEqual(result.status, "timed_out")
        self.assertTrue(result.timed_out)
        self.assertLess(elapsed, 12.0)
        inspected = subprocess.run(
            ["docker", "inspect", result.container_name],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(inspected.returncode, 0)

    def test_excessive_program_output_is_truncated_and_terminated(self) -> None:
        result = DockerScientificSandbox.load_default().execute(
            language="python",
            source="import sys\nsys.stdout.write('x' * 70000)\n",
            stdin_text="",
        )

        self.assertEqual(result.status, "output_limited")
        self.assertTrue(result.output_limited)
        self.assertEqual(len(result.stdout_bytes), 64 * 1024)

    def test_general_image_executes_each_declared_language(self) -> None:
        programs = {
            "python": "print(42)\n",
            "javascript": "console.log(42);\n",
            "typescript": "const value: number = 42; console.log(value);\n",
            "c": "#include <stdio.h>\nint main(void) { puts(\"42\"); return 0; }\n",
            "cpp": "#include <iostream>\nint main() { std::cout << 42 << '\\n'; }\n",
            "c_sharp": "using System; class Program { static void Main() { Console.WriteLine(42); } }\n",
            "java": "public class Main { public static void main(String[] a) { System.out.println(42); } }\n",
            "go": "package main\nimport \"fmt\"\nfunc main() { fmt.Println(42) }\n",
            "rust": "fn main() { println!(\"42\"); }\n",
            "php": "<?php echo \"42\\n\";\n",
            "ruby": "puts 42\n",
            "kotlin": "fun main() { println(42) }\n",
            "bash": "printf '42\\n'\n",
        }
        sandbox = DockerScientificSandbox.load_default()
        for language, source in programs.items():
            with self.subTest(language=language):
                result = sandbox.execute(
                    language=language,
                    source=source,
                    stdin_text="",
                )
                self.assertEqual(
                    (result.status, result.stdout_text, result.stderr_text),
                    ("passed", "42\n", ""),
                )

    def test_swift_image_compiles_and_executes_under_the_same_policy(self) -> None:
        result = DockerScientificSandbox.load_default().execute(
            language="swift",
            source="print(42)\n",
            stdin_text="",
        )

        self.assertEqual(
            (result.status, result.stdout_text, result.stderr_text),
            ("passed", "42\n", ""),
        )
        self.assertEqual(result.attestation.network_mode, "none")
        self.assertTrue(result.attestation.read_only_rootfs)
        self.assertEqual(result.attestation.mount_count, 0)

    def test_assertions_cannot_silently_pass_in_compiled_languages(self) -> None:
        fixtures = {
            "java": (
                "public class Main { public static void main(String[] a) { "
                "assert false : \"fixture\"; } }\n"
            ),
            "kotlin": "fun main() { assert(false) { \"fixture\" } }\n",
            "swift": "assert(false, \"fixture\")\n",
        }
        sandbox = DockerScientificSandbox.load_default()
        for language, source in fixtures.items():
            with self.subTest(language=language):
                result = sandbox.execute(
                    language=language,
                    source=source,
                    stdin_text="",
                )
                self.assertEqual(result.status, "runtime_failed")
                self.assertNotEqual(result.exit_code, 0)

    def test_javascript_console_assertion_emits_failure_evidence(self) -> None:
        result = DockerScientificSandbox.load_default().execute(
            language="javascript",
            source="console.assert(false, 'fixture');\n",
            stdin_text="",
        )

        self.assertEqual(result.status, "passed")
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Assertion failed", result.stderr_text)

    def test_c_sharp_debug_assertion_emits_failure_evidence(self) -> None:
        result = DockerScientificSandbox.load_default().execute(
            language="c_sharp",
            source=(
                "using System; using System.Diagnostics; "
                "class Program { static void Main() { "
                "Debug.Listeners.Add(new TextWriterTraceListener(Console.Error)); "
                "Debug.AutoFlush = true; Debug.Assert(false, \"fixture\"); } }\n"
            ),
            stdin_text="",
        )

        self.assertEqual(result.status, "passed")
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Fail: fixture", result.stderr_text)

    def test_go_test_mode_runs_the_real_test_runner_with_pinned_dependencies(self) -> None:
        source = (
            "package main\n"
            "import (\n"
            "  \"testing\"\n"
            "  \"github.com/stretchr/testify/assert\"\n"
            ")\n"
            "func TestFixture(t *testing.T) { assert.Equal(t, 42, 42) }\n"
        )
        result = DockerScientificSandbox.load_default().execute(
            language="go",
            mode="test",
            source=source,
            stdin_text="",
        )

        self.assertEqual(result.status, "passed")
        self.assertEqual(result.stderr_text, "")

    def test_go_test_mode_reports_a_failed_assertion(self) -> None:
        source = (
            "package main\n"
            "import \"testing\"\n"
            "func TestFixture(t *testing.T) { t.Fatal(\"fixture\") }\n"
        )
        result = DockerScientificSandbox.load_default().execute(
            language="go",
            mode="test",
            source=source,
            stdin_text="",
        )

        self.assertEqual(result.status, "runtime_failed")

    def test_rust_test_mode_runs_the_real_test_runner_with_pinned_dependencies(self) -> None:
        source = (
            "#[cfg(test)]\n"
            "mod tests {\n"
            "    use regex::Regex;\n"
            "    #[test]\n"
            "    fn fixture() {\n"
            "        let _ = rand::random::<u8>();\n"
            "        assert!(Regex::new(\"^ok$\").unwrap().is_match(\"ok\"));\n"
            "        assert_eq!(format!(\"{:x}\", md5::compute(b\"a\")).len(), 32);\n"
            "    }\n"
            "}\n"
        )
        result = DockerScientificSandbox.load_default().execute(
            language="rust",
            mode="test",
            source=source,
            stdin_text="",
        )

        self.assertEqual(result.status, "passed")
        self.assertEqual(result.stderr_text, "")

    def test_rust_test_mode_reports_a_failed_assertion(self) -> None:
        source = (
            "#[cfg(test)]\n"
            "mod tests {\n"
            "    #[test]\n"
            "    fn fixture() { assert!(false, \"fixture\"); }\n"
            "}\n"
        )
        result = DockerScientificSandbox.load_default().execute(
            language="rust",
            mode="test",
            source=source,
            stdin_text="",
        )

        self.assertEqual(result.status, "runtime_failed")

    def test_bash_image_includes_the_mceval_bc_dependency(self) -> None:
        result = DockerScientificSandbox.load_default().execute(
            language="bash",
            source="test \"$(printf '6 * 7\\n' | bc)\" = 42\n",
            stdin_text="",
        )

        self.assertEqual(result.status, "passed")
        self.assertEqual(result.stderr_text, "")


if __name__ == "__main__":
    unittest.main()
