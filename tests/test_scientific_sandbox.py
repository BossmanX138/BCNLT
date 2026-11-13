from __future__ import annotations

import subprocess
import time
import unittest

from bosscrafts_model.scientific_sandbox import DockerScientificSandbox


class ScientificSandboxIntegrationTests(unittest.TestCase):
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
            "sha256:153a93c2ca76aacacdbdb7d84d4e1e63767c46ddd7e42f9957e9fe7f7c782f32",
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


if __name__ == "__main__":
    unittest.main()
