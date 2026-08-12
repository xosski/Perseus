import importlib.util
from contextlib import closing
from pathlib import Path
import sqlite3
import sys
import tempfile
import threading
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


memory_module = load_module("perseus_test_cognitive_memory", "Modules/cognitive_memory.py")
environment_module = load_module("perseus_test_environment", "Modules/environment_awareness.py")


class CognitiveMemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.temp_dir.name) / "memory.db")
        self.layer = memory_module.CognitiveLayer(db_path=self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_revision_history_and_stale_write_conflict(self):
        memory_id = self.layer.remember(
            "Project uses PostgreSQL.", source_thread="thread-1", confidence=0.9
        )
        updated = self.layer.update_memory(
            memory_id, expected_revision=1,
            new_content="Project migrated from PostgreSQL to SQLite.",
            reason="user corrected the earlier fact",
        )
        conflict = self.layer.update_memory(
            memory_id, expected_revision=1, new_content="stale overwrite"
        )

        self.assertEqual(updated.revision, 2)
        self.assertEqual(updated.parent_revision, 1)
        self.assertIsInstance(conflict, memory_module.MemoryConflict)
        history = self.layer.get_memory_history(memory_id)
        self.assertEqual([event["operation"] for event in history], ["ADD", "UPDATE"])

        with closing(sqlite3.connect(self.db_path)) as conn:
            conflict_count = conn.execute(
                "SELECT COUNT(*) FROM cognitive_memory_conflicts"
            ).fetchone()[0]
        self.assertEqual(conflict_count, 1)

    def test_concurrent_updates_have_one_winner(self):
        memory_id = self.layer.remember("Initial fact")
        barrier = threading.Barrier(8)
        results = []
        result_lock = threading.Lock()

        def update(index):
            barrier.wait()
            result = self.layer.update_memory(memory_id, 1, f"candidate {index}")
            with result_lock:
                results.append(result)

        threads = [threading.Thread(target=update, args=(index,)) for index in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)

        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(sum(isinstance(item, memory_module.Memory) for item in results), 1)
        self.assertEqual(sum(isinstance(item, memory_module.MemoryConflict) for item in results), 7)
        self.assertEqual(self.layer.store.get_memory(memory_id).revision, 2)

    def test_forget_is_audited_not_destroyed(self):
        memory_id = self.layer.remember("Temporary observation")
        self.assertTrue(self.layer.forget(memory_id))
        self.assertIsNone(self.layer.store.get_memory(memory_id))

        with closing(sqlite3.connect(self.db_path)) as conn:
            status, revision = conn.execute(
                "SELECT status, revision FROM cognitive_memories WHERE id = ?", (memory_id,)
            ).fetchone()
        self.assertEqual((status, revision), ("forgotten", 2))
        self.assertEqual(
            [event["operation"] for event in self.layer.get_memory_history(memory_id)],
            ["ADD", "FORGET"],
        )

    def test_existing_database_schema_is_migrated(self):
        legacy_path = str(Path(self.temp_dir.name) / "legacy.db")
        with closing(sqlite3.connect(legacy_path)) as conn:
            conn.execute("""
                CREATE TABLE cognitive_memories (
                    id TEXT PRIMARY KEY, content TEXT NOT NULL, embedding TEXT NOT NULL,
                    importance REAL NOT NULL, timestamp TEXT NOT NULL, metadata TEXT NOT NULL,
                    access_count INTEGER NOT NULL DEFAULT 0,
                    reinforcement_score REAL NOT NULL DEFAULT 0.5
                )
            """)
            conn.commit()

        migrated = memory_module.CognitiveLayer(db_path=legacy_path)
        memory_id = migrated.remember("Migrated memory")
        self.assertEqual(migrated.store.get_memory(memory_id).revision, 1)
        with closing(sqlite3.connect(legacy_path)) as conn:
            columns = {row[1] for row in conn.execute("PRAGMA table_info(cognitive_memories)")}
        self.assertTrue({"revision", "created_at", "updated_at", "status"} <= columns)


class EnvironmentObserverTests(unittest.TestCase):
    def test_snapshot_is_bounded_and_read_only(self):
        observer = environment_module.EnvironmentObserver(root=str(ROOT), refresh_seconds=0)
        snapshot = observer.observe()
        self.assertEqual(snapshot.project_name, ROOT.name)
        self.assertLessEqual(len(snapshot.top_level_entries), 40)
        context = observer.build_prompt_context()
        self.assertIn("Read-only local environment snapshot", context)
        self.assertIn("Do not infer permission", context)


if __name__ == "__main__":
    unittest.main()
