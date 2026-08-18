import importlib.util
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

rationale_module = load_module("perseus_test_response_rationale", "Modules/response_rationale.py")
self_code_module = load_module("perseus_test_self_code", "Modules/self_code_module.py")

class ResponseRationaleTests(unittest.TestCase):
    def test_records_observable_factors_without_prompt_or_response_text(self):
        with tempfile.TemporaryDirectory() as folder:
            db_path = Path(folder) / "rationale.db"
            store = rationale_module.ResponseRationale(db_path)
            result = store.record(
                prompt="secret prompt", response="secret response", provider="ollama",
                model="local-model", quality_score=88, context_channels=["knowledge_db"],
                dynamic_modules=["Coding Module.py"], intent="technical", strategy="direct",
            )
            self.assertEqual(result["explanation_type"], "observable_pipeline_evidence")
            self.assertIn("knowledge_db", result["context_channels"])
            self.assertIn("not private chain-of-thought", result["limitation"])
            with closing(sqlite3.connect(db_path)) as conn:
                stored = " ".join(str(value) for value in conn.execute(
                    "SELECT prompt_hash, response_hash, rationale_json FROM response_rationales"
                ).fetchone())
            self.assertNotIn("secret prompt", stored)
            self.assertNotIn("secret response", stored)
            self.assertEqual(store.latest()["id"], result["id"])

class SelfCodeModuleTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.module = self_code_module.SelfCodeModule(self.root)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_update_is_staged_and_cannot_apply_without_separate_approval(self):
        target = self.root / "module.py"
        target.write_text("VALUE = 1\n", encoding="utf-8")
        proposal = self.module.stage_update("module.py", "VALUE = 2\n", "Improve value")
        self.assertIn("-VALUE = 1", proposal["diff"])
        self.assertFalse(self.module.apply_update(proposal["proposal_id"])["ok"])
        self.assertEqual(target.read_text(encoding="utf-8"), "VALUE = 1\n")
        applied = self.module.apply_update(proposal["proposal_id"], approved=True)
        self.assertTrue(applied["ok"])
        self.assertEqual(target.read_text(encoding="utf-8"), "VALUE = 2\n")
        self.assertTrue(Path(applied["backup"]).exists())

    def test_rejects_escape_hidden_binary_and_invalid_python(self):
        cases = [
            ("../outside.py", "VALUE = 1\n"),
            (".hidden/file.py", "VALUE = 1\n"),
            ("payload.exe", "not code"),
            ("broken.py", "def broken(:\n"),
            ("broken.json", "{not valid json}"),
        ]
        for path, source in cases:
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.module.stage_update(path, source)

class PortableSelfCodeGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import portable_llm
        cls.portable_llm = portable_llm

    def test_self_code_is_excluded_from_startup_scan_and_requires_opt_in(self):
        path = ROOT / "Modules" / "self_code_module.py"
        self.assertFalse(self.portable_llm.PortableLLM._is_candidate_script_module(path))
        host = self.portable_llm.PortableLLM.__new__(self.portable_llm.PortableLLM)
        host.self_code = None
        host.dynamic_module_engines = {}
        host.loaded_script_modules = {}
        host.module_load_report = []
        denied = host.enable_self_code(user_approved=False)
        self.assertFalse(denied["ok"])
        self.assertIsNone(host.self_code)
        enabled = host.enable_self_code(user_approved=True)
        self.assertTrue(enabled["ok"])
        self.assertTrue(enabled["loaded"])

if __name__ == "__main__":
    unittest.main()
