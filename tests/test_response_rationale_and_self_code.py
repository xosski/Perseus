import importlib.util
from contextlib import closing
import json
import os
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
introspection_module = load_module("perseus_test_introspection", "Modules/Introspective Learning.py")
search_module = load_module("perseus_test_search", "Modules/Search Augmentation.py")

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

class GroundedContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import portable_llm
        cls.portable_llm = portable_llm

    def test_grounding_selects_relevant_environment_facts(self):
        context = (
            "Read-only environment: - Local date and time: Tuesday, August 18, 2026 at 7:15 AM "
            "(Pacific Daylight Time, UTC-07:00) - Project: Perseus - Git branch: main | "
            "ONLINE SEARCH CONTEXT Internal evidence only. Search query: current local time "
            "Source: example.invalid Retrieved: just now Snippet: raw payload"
        )
        facts = self.portable_llm._relevant_context_facts("Give me the current local time", context)
        self.assertTrue(facts)
        self.assertIn("Local date and time", facts[0])
        answer = self.portable_llm.PortableLLM._build_grounded_response(
            "Give me the current local time", context
        )
        self.assertIn("Tuesday, August 18, 2026", answer)
        self.assertNotIn("I found relevant learned context", answer)
        self.assertNotIn("ONLINE SEARCH CONTEXT", answer)
        self.assertNotIn("raw payload", answer)
        enriched_prompt = f"RAW_CONTEXT_DO_NOT_OUTPUT_BEGIN\n{context}\nRAW_CONTEXT_DO_NOT_OUTPUT_END"
        sanitized = self.portable_llm._sanitize_visible_response(answer, prompt_payload=enriched_prompt)
        self.assertIn("Tuesday, August 18, 2026", sanitized)
        self.assertIn("Local date and time", sanitized)

    def test_grounding_rejects_unrelated_context(self):
        context = "- Git branch: main - Project: Perseus"
        self.assertEqual(
            self.portable_llm._relevant_context_facts("How deep is the ocean?", context), []
        )

    def test_search_fallback_handles_nested_transformation_ambiguity_generally(self):
        context = """ONLINE SEARCH CONTEXT
1. Sparkling water - reference article
   Source: reference.example
   URL: https://reference.example/sparkling-water
   Retrieved: 2026-08-18T00:00:00Z
   Snippet: Sparkling water is water containing dissolved carbon dioxide gas. It differs from still water because of its carbonation.

Instruction: use relevant evidence only.
"""
        answer = self.portable_llm._answer_from_online_search_context(
            context, "How can someone convert sparkling water into water?"
        )
        self.assertIn("no distinct conversion is defined", answer)
        self.assertIn("sparkling water", answer.lower())
        self.assertIn("name that endpoint", answer)
        self.assertIn("reference.example", answer)
        self.assertNotIn("ONLINE SEARCH CONTEXT", answer)
        self.assertNotIn("Snippet:", answer)

    def test_broad_category_transformation_requests_require_a_defined_source(self):
        answer = self.portable_llm._portable_fallback_response(
            "How can someone convert water into sparkling water?"
        )
        self.assertIn("water names a broad category", answer)
        self.assertIn("sparkling water is one specific kind", answer)
        self.assertIn("exact starting material", answer)

    def test_chat_memory_is_not_treated_as_factual_ingested_evidence(self):
        context = """Ingested context:
Source: chat-memory/technical/example
Learned chat interaction: Memory categories: task Memory summary: user mentioned auroras.

Source: science/reference
Auroras occur when charged particles interact with gases in the upper atmosphere.

Output requirements:
Answer directly.
"""
        sources = self.portable_llm._parse_ingested_context_sources(context)
        self.assertEqual([source["title"] for source in sources], ["science/reference"])
        points = self.portable_llm._synthesize_ingested_points(
            sources, "Why do auroras occur?"
        )
        self.assertTrue(points)
        self.assertNotIn("chat-memory", " ".join(points))
        self.assertNotIn("Memory categories", " ".join(points))

    def test_search_context_parser_keeps_blank_line_separated_sources_distinct(self):
        results = [
            {
                "title": "First technical source",
                "url": "https://first.example/topic",
                "source": "search",
                "snippet": "The first source explains a technical mechanism using enough detail to be useful.",
            },
            {
                "title": "Second technical source",
                "url": "https://second.example/topic",
                "source": "search",
                "snippet": "The second source supplies a separate supporting explanation for the same topic.",
            },
        ]
        context = self.portable_llm._format_online_search_context(results, "technical topic")
        parsed = self.portable_llm._parse_online_search_results(context)
        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[0]["url"], "https://first.example/topic")
        self.assertEqual(parsed[1]["url"], "https://second.example/topic")
        self.assertNotIn("Second technical source", parsed[0]["snippet"])

    def test_search_synthesis_rejects_page_chrome_and_keeps_relevant_evidence(self):
        results = [
            {
                "title": "Aurora - encyclopedia",
                "url": "https://science.example/aurora",
                "source": "search",
                "snippet": (
                    "Jump to content Main menu move to sidebar create account log in privacy policy. "
                    "Auroras occur when charged particles interact with gases in a planet's upper atmosphere."
                ),
            },
            {
                "title": "Unrelated account page",
                "url": "https://other.example/account",
                "source": "search",
                "snippet": "Create an account to manage preferences and receive site announcements.",
            },
        ]
        points = self.portable_llm._synthesize_lookup_points(results, "What causes auroras?")
        self.assertEqual(len(points), 1)
        self.assertIn("charged particles", points[0])
        self.assertNotIn("Main menu", points[0])

    def test_search_synthesis_matches_minor_topic_typo_to_explanatory_evidence(self):
        results = [
            {
                "title": "Photosynthesis - encyclopedia",
                "url": "https://science.example/photosynthesis",
                "source": "search",
                "snippet": (
                    "Photosynthesis Page excerpt: Photosynthesis converts light energy into chemical energy "
                    "stored in sugars. The process was investigated by several researchers during the "
                    "nineteenth century."
                ),
            }
        ]
        points = self.portable_llm._synthesize_lookup_points(
            results, "How does photosynhesis convert light into chemical energy?"
        )
        self.assertTrue(points)
        self.assertIn("converts light energy", points[0])
        self.assertNotIn("Page excerpt", " ".join(points))

    def test_repeated_corrected_titles_outrank_unrelated_exact_typo_token(self):
        results = [
            {
                "title": "Astronomy - encyclopedia",
                "url": "https://science.example/astronomy",
                "source": "search",
                "snippet": "Astronomy studies celestial objects, space, and the physical universe as a whole.",
            },
            {
                "title": "Introduction to Astronomy",
                "url": "https://university.example/astronomy",
                "source": "search",
                "snippet": "Astronomy uses observation and physics to explain stars, planets, and galaxies.",
            },
            {
                "title": "Astro loyalty rewards",
                "url": "https://shopping.example/astro",
                "source": "search",
                "snippet": "Customers can join the Astro rewards program to receive shopping discounts.",
            },
        ]
        points = self.portable_llm._synthesize_lookup_points(
            results, "How does astro nomie work?"
        )
        self.assertTrue(points)
        self.assertIn("celestial objects", " ".join(points))
        self.assertNotIn("rewards", " ".join(points).lower())

    def test_search_synthesis_matches_split_misspelled_compound_term(self):
        results = [
            {
                "title": "Microscopy - encyclopedia",
                "url": "https://science.example/microscopy",
                "source": "search",
                "snippet": (
                    "Microscopy uses microscopes to view objects and structures that cannot be resolved "
                    "with the unaided eye. Different instruments use light or electron beams."
                ),
            }
        ]
        points = self.portable_llm._synthesize_lookup_points(
            results, "How does micro scopi work?"
        )
        self.assertTrue(points)
        self.assertIn("uses microscopes", points[0])

    def test_compound_request_composes_environment_and_search_answers(self):
        request = (
            "Hello Perseus can you give me the time and explain how to tturn "
            "sparkling water into jsut water"
        )
        parts = self.portable_llm._split_compound_requests(request)
        self.assertEqual(len(parts), 2)
        self.assertIn("turn sparkling water into just water", parts[1])
        context = """Local date and time: Tuesday, August 18, 2026 at 7:30 AM (PDT, UTC-07:00)
ONLINE SEARCH CONTEXT
1. Sparkling water - reference article
   Source: reference.example
   URL: https://reference.example/sparkling-water
   Retrieved: 2026-08-18T00:00:00Z
   Snippet: Sparkling water is water containing dissolved carbon dioxide gas.

Instruction: use relevant evidence only.
"""
        answer = self.portable_llm._answer_compound_request(context, request)
        self.assertIn("1. Local date and time", answer)
        self.assertIn("2. As worded, no distinct conversion is defined", answer)
        self.assertIn("sparkling water is already a kind of water", answer.lower())
        self.assertNotIn("Working directory", answer)
        self.assertNotIn("ONLINE SEARCH CONTEXT", answer)

    def test_compound_parser_tracks_each_request_and_ignores_address_words(self):
        request = (
            "Hey Perseus can you give me the time and explain how to tturn "
            "sparkling water into jsut water"
        )
        host = self.portable_llm.PortableLLM.__new__(self.portable_llm.PortableLLM)
        profile = host._profile_prompt(request)
        packet = host._build_parser_packet(request, profile)
        self.assertTrue(packet["question"])
        self.assertEqual(len(packet["subrequests"]), 2)
        self.assertIn("sparkling", packet["focus_terms"])
        self.assertNotIn("perseus", packet["focus_terms"])
        self.assertNotIn("tturn", packet["focus_terms"])
        self.assertNotIn("jsut", packet["focus_terms"])

    def test_search_enrichment_evaluates_compound_clauses_independently(self):
        class Decision:
            def __init__(self, should_search):
                self.should_search = should_search
                self.reason = "factual clause"

        class Searcher:
            def __init__(self):
                self.queries = []

            def should_search(self, query, local_context=""):
                self.queries.append((query, local_context))
                return Decision("sparkling water" in query.lower())

            def search_and_build_context(self, query):
                return "ONLINE SEARCH CONTEXT\nQuery: " + query

        searcher = Searcher()
        host = self.portable_llm.PortableLLM.__new__(self.portable_llm.PortableLLM)
        host.search_augmentation = searcher
        original = self.portable_llm.EnrichedPrompt(
            text="request", has_context=True, context_preview="Local date and time: now"
        )
        result = host._enrich_prompt_with_online_search(
            original,
            "Give me the time and explain how to tturn sparkling water into jsut water",
        )
        self.assertEqual(len(searcher.queries), 2)
        self.assertIn("Local date and time", searcher.queries[0][1])
        self.assertEqual(searcher.queries[1][1], "")
        self.assertIn("turn sparkling water into just water", searcher.queries[1][0])
        self.assertIn("ONLINE SEARCH CONTEXT", result.text)

    def test_misspelled_factual_request_searches_past_unrelated_local_context(self):
        class Decision:
            should_search = True
            reason = "unresolved factual request"

        class Searcher:
            def __init__(self):
                self.local_context = None

            def should_search(self, _query, local_context=""):
                self.local_context = local_context
                return Decision()

            def search_and_build_context(self, query):
                return "ONLINE SEARCH CONTEXT\nQuery: " + query

        searcher = Searcher()
        host = self.portable_llm.PortableLLM.__new__(self.portable_llm.PortableLLM)
        host.search_augmentation = searcher
        original = self.portable_llm.EnrichedPrompt(
            text="request",
            has_context=True,
            context_preview="Project: Perseus | Working directory: C:\\workspace | Git branch: main",
        )
        result = host._enrich_prompt_with_online_search(
            original, "How does photosynhesis convert light into chemical energy?"
        )
        self.assertEqual(searcher.local_context, "")
        self.assertIn("ONLINE SEARCH CONTEXT", result.text)

    def test_misspelled_explanation_form_is_a_factual_search_request(self):
        searcher = search_module.SearchAugmentation.__new__(search_module.SearchAugmentation)
        decision = searcher.should_search(
            "How does photosynhesis convert light into chemical energy?", local_context=""
        )
        self.assertTrue(decision.should_search)

    def test_underspecified_followup_inherits_previous_topic_before_retrieval(self):
        class Message:
            role = "user"
            content = "How does micro scopi work?"
            metadata = {}

        resolved = self.portable_llm._resolve_contextual_followup(
            "Can you provide more information?", [Message()]
        )
        self.assertEqual(resolved, "Explain micro scopi in more detail.")

        explicit = self.portable_llm._resolve_contextual_followup(
            "Can you provide more information about radio telescopes?", [Message()]
        )
        self.assertEqual(explicit, "Can you provide more information about radio telescopes?")

    def test_deeper_explanation_requests_search_even_with_local_context(self):
        searcher = search_module.SearchAugmentation.__new__(search_module.SearchAugmentation)
        decision = searcher.should_search(
            "Explain micro scopi in more detail.",
            local_context="Microscopy uses lenses to enlarge an image.",
        )
        self.assertTrue(decision.should_search)

    def test_standalone_typo_reaches_portable_fallback_without_external_provider(self):
        prompt = "Hey Perseus hwo do I turn sparkling water into jsut water"
        answer = self.portable_llm._portable_fallback_response(prompt)
        self.assertIn("sparkling water is already a kind of water", answer.lower())
        self.assertNotIn("No response generated", answer)
        self.assertTrue(self.portable_llm._response_covers_request(prompt, answer))
        self.assertFalse(
            self.portable_llm._response_covers_request(
                "Give me the current time and explain why auroras happen",
                "The current time is 7:30 AM.",
            )
        )

        class ImportedFallback:
            available = True

            def generate(self, *_args, **_kwargs):
                raise AssertionError("portable fallback must not delegate enriched prompts")

        class Manager:
            providers = {"fallback": ImportedFallback()}

        host = self.portable_llm.PortableLLM.__new__(self.portable_llm.PortableLLM)
        host.manager = Manager()
        generated = host._generate_with_provider(
            "fallback", prompt, profile=None, refine=False, prior_response=None
        )
        self.assertEqual(generated, answer)

        hidden_source = f"Retrieved source code:\n{answer}"
        self.assertEqual(
            self.portable_llm._sanitize_visible_response(answer, prompt_payload=hidden_source),
            "",
        )
        self.assertEqual(
            self.portable_llm._sanitize_selected_response(
                answer, prompt_payload=hidden_source, provider="fallback"
            ),
            answer,
        )

    def test_final_prompt_request_marker_wins_over_nested_context(self):
        enriched = (
            "Current prompt payload:\nold internal text\n"
            "User request to answer:\nExplain why auroras happen"
        )
        self.assertEqual(
            self.portable_llm._extract_user_request(enriched),
            "Explain why auroras happen",
        )

    def test_compound_clause_triggers_factual_search_without_local_context(self):
        searcher = search_module.SearchAugmentation.__new__(search_module.SearchAugmentation)
        decision = searcher.should_search(
            "Hey Perseus how do I turn sparkling water into water", local_context=""
        )
        self.assertTrue(decision.should_search)

    def test_language_analysis_is_guidance_not_grounding_evidence(self):
        class LanguageEngine:
            def build_prompt_context(self, _prompt):
                return "Intent analysis mentioning the user's words, but containing no external facts."

        host = self.portable_llm.PortableLLM.__new__(self.portable_llm.PortableLLM)
        host.english_language_engine = LanguageEngine()
        result = host._enrich_prompt_with_language_engine("Explain an unknown fact")
        self.assertFalse(result.has_context)
        self.assertEqual(result.context_preview, "")
        self.assertIn("Intent analysis", result.text)

    def test_small_talk_does_not_activate_environment_grounding(self):
        class Observer:
            def build_prompt_context(self, _prompt):
                raise AssertionError("small talk should not query environment context")

        host = self.portable_llm.PortableLLM.__new__(self.portable_llm.PortableLLM)
        host.environment_observer = Observer()
        original = self.portable_llm.EnrichedPrompt(text="how are you", has_context=False)
        result = host._enrich_prompt_with_environment(original, "how are you")
        self.assertIs(result, original)
        self.assertFalse(result.has_context)


class IntrospectionCoverageTests(unittest.TestCase):
    def test_partial_compound_answer_is_not_marked_complete(self):
        engine = introspection_module.IntrospectiveLearning.__new__(
            introspection_module.IntrospectiveLearning
        )
        critique = engine._critique_response(
            "Hey Perseus, give me the current time and explain why auroras happen",
            "The local time is 7:30 AM.",
        )
        self.assertFalse(critique.answered_question)
        self.assertIn("actual question", " ".join(critique.issues))


class PortableSelfCodeGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import portable_llm
        cls.portable_llm = portable_llm

    def test_session_provider_configuration_does_not_persist_api_key(self):
        class Provider:
            available = True

        class Manager:
            def __init__(self):
                self.providers = {"openai": Provider()}
                self.switched = None

            def reload_providers(self):
                self.seen_key = os.environ.get("OPENAI_API_KEY")
                self.seen_url = os.environ.get("OPENAI_BASE_URL")
                return ["openai"]

            def switch_provider(self, conversation_id, provider, model):
                self.switched = (conversation_id, provider, model)
                return True

        class Conversation:
            id = "conversation-id"
            provider = "fallback"
            model = "fallback"

        old_key = os.environ.pop("OPENAI_API_KEY", None)
        old_url = os.environ.pop("OPENAI_BASE_URL", None)
        try:
            host = self.portable_llm.PortableLLM.__new__(self.portable_llm.PortableLLM)
            host.manager = Manager()
            host.conversation = Conversation()
            host.strict_local_only = True
            host.provider = "fallback"
            host.model = "fallback"
            result = host.configure_session_provider(
                "openai", api_key="session-secret", model="test-model", endpoint="https://llm.example/v1"
            )
            self.assertTrue(result["ok"])
            self.assertEqual(host.manager.seen_key, "session-secret")
            self.assertEqual(host.manager.seen_url, "https://llm.example/v1")
            self.assertNotIn("session-secret", json.dumps(result))
            self.assertNotIn("OPENAI_API_KEY", os.environ)
            self.assertNotIn("OPENAI_BASE_URL", os.environ)
            self.assertFalse(host.strict_local_only)
            self.assertEqual(host.manager.switched, ("conversation-id", "openai", "test-model"))
        finally:
            if old_key is not None:
                os.environ["OPENAI_API_KEY"] = old_key
            if old_url is not None:
                os.environ["OPENAI_BASE_URL"] = old_url

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
