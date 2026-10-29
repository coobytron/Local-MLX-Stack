import unittest
from local_mlx_stack.routing import select

class RoutingTests(unittest.TestCase):
    def test_default_fast(self):
        self.assertEqual(select("Summarize this document").role, "fast")

    def test_code_precedes_reasoning(self):
        self.assertEqual(select("Plan a Swift shader implementation").role, "coder")

    def test_reasoner(self):
        self.assertEqual(select("Compare two system architectures").role, "reasoner")

    def test_manual_override(self):
        self.assertEqual(select("Summarize", "reasoner").role, "reasoner")
        with self.assertRaises(ValueError):
            select("Hi", "external")
