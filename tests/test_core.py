import tempfile
import unittest
from pathlib import Path
from local_mlx_stack.core import Stack
from local_mlx_stack.ollama import Settings
from local_mlx_stack.acb import guidance, checklist_path, locate

class FakeClient:
    def __init__(self):
        self.settings = Settings()
        self.calls = []

    def chat(self, role, prompt, system=""):
        self.calls.append((role, prompt, system))
        return f"response for {role}"

class CoreTests(unittest.TestCase):
    def test_one_model_by_default(self):
        client = FakeClient()
        answer = Stack(client).ask("Explain a color-space conversion")
        self.assertEqual(answer.role, "fast")
        self.assertEqual([x[0] for x in client.calls], ["fast"])

    def test_pipeline_explicit_trigger(self):
        client = FakeClient()
        stack = Stack(client)
        with self.assertRaises(ValueError):
            stack.pipeline("Implement a feature", "none")
        self.assertEqual(client.calls, [])
        result = stack.pipeline("Implement a feature", "review")
        self.assertEqual([x[0] for x in client.calls], ["fast", "reasoner", "coder", "reasoner"])
        self.assertIn("review", result)
        self.assertEqual(result["trigger"], "review")

    def test_other_trigger_does_not_add_review(self):
        client = FakeClient()
        result = Stack(client).pipeline("Task", "tools")
        self.assertNotIn("review", result)
        self.assertEqual(len(client.calls), 3)

    def test_acb_checks_are_optional(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "AGENTS.md").write_text("Proof rules")
            self.assertIn("Proof rules", guidance("model stack", "reasoner", p))
            self.assertIn("AI-SPECIALIST", checklist_path("MLX model stack", "reasoner"))
