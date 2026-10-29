import tempfile
import unittest
from pathlib import Path
from local_mlx_stack.project import collect_project


class ProjectTests(unittest.TestCase):
    def test_grounded_context_and_secret_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / 'docs').mkdir()
            (p / '.venv').mkdir()
            (p / 'README.md').write_text('Existing FastAPI React bridge')
            (p / 'docs' / 'ARCHITECTURE.md').write_text('Draw Things HTTP and gRPC adapters')
            (p / 'docs' / 'LOCAL-LANGUAGE-MODELS.md').write_text('Already supports Ollama providers')
            (p / 'secrets.json').write_text('NEVER_SHARE_ME')
            (p / '.env').write_text('NEVER_SHARE_ENV')
            (p / 'AGENTS.md').write_text('NEVER_SHARE_AGENT_RULES')
            (p / '.venv' / 'hidden.py').write_text('NEVER_SHARE_VENV')
            snapshot = collect_project(p, 'Design local Draw Things controller')
            self.assertIn('FastAPI React bridge', snapshot.text)
            self.assertIn('Already supports Ollama providers', snapshot.text)
            self.assertIn('docs/ARCHITECTURE.md', snapshot.files)
            for forbidden in ('NEVER_SHARE_ME', 'NEVER_SHARE_ENV', 'NEVER_SHARE_AGENT_RULES', 'NEVER_SHARE_VENV'):
                self.assertNotIn(forbidden, snapshot.text)

    def test_no_symlink_escape(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as outside:
            base = Path(tmp)
            (Path(outside) / 'bad.py').write_text('NEVER_SHARE_EXTERNAL')
            (base / 'external').symlink_to(outside, target_is_directory=True)
            (base / 'README.md').write_text('safe')
            snapshot = collect_project(base, 'test')
            self.assertNotIn('NEVER_SHARE_EXTERNAL', snapshot.text)

    def test_nonexistent_directory(self):
        with self.assertRaisesRegex(ValueError, 'Project directory not found'):
            collect_project(Path('/there/is/no/such/repository'), 'task')

    def test_text_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            (p/'README.md').write_text('A'*50000)
            snapshot=collect_project(p,'controller',max_chars=2200)
            self.assertLess(len(snapshot.text), 2600)
            self.assertIn('TRUNCATED', snapshot.text)
