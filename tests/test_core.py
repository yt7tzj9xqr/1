import json
from pathlib import Path
import tempfile
import unittest

from reportbench_mm.cache import JsonCache
from reportbench_mm.dataset import load_tasks, stratified_subset
from reportbench_mm.providers.openalex import compact_query, extract_cutoff, filter_papers, normalize_title, search_queries
from reportbench_mm.schemas import Paper
from reportbench_mm.pipelines.rag import (
    anchor_coverage, keywords, matches_anchor_phrase, normalize_source_citations,
    sanitize_report, score_paper, select_writing_papers, writing_score,
)
from reportbench_mm.config import Settings
from reportbench_mm.providers.composite import CompositeScholarProvider
from reportbench_mm.evaluation.reference import maximum_matches, normalize_url, title_match
from reportbench_mm.evaluation.statements import cited_statements, extract_noncited
from reportbench_mm.evaluation.aggregate import aggregate
from reportbench_mm.models.minimax import MiniMaxClient
from reportbench_mm.retrieval import (
    adaptive_search, canonical_search_title, diverse_top_papers, is_scholarly_candidate, parallel_search,
    model_rerank_papers, plan_search_queries,
)
from reportbench_mm.web_reader import arxiv_pdf_url, extract_pdf_text, parse_academic_html
from reportbench_mm.providers.minimax_search import MiniMaxSearchProvider
from reportbench_mm.prompts import (
    _repair_output_is_usable, generated_report_is_usable, repair_grounded_report,
)


class CoreTests(unittest.TestCase):
    def test_report_quality_gate_requires_length_and_sources(self):
        sources = " ".join(f"https://example.org/{index}" for index in range(4))
        self.assertTrue(generated_report_is_usable(("evidence " * 350) + sources))
        self.assertFalse(generated_report_is_usable("conclusion only"))
        self.assertFalse(generated_report_is_usable("uncited " * 400))

    def test_grounding_repair_requires_atomic_cited_sentences(self):
        class Model:
            def generate(self, messages, **kwargs):
                self.prompt = messages[0]["content"]
                return "Supported claim (https://example.org/p)."

        model = Model()
        paper = Paper("p", "Paper Title", 2020, "https://example.org/p", "Direct evidence for claim.")
        report = repair_grounded_report(
            "Unsupported draft.", [paper], model, "test", "600-800", "10-12",
        )
        self.assertIn("exactly one URL", model.prompt)
        self.assertIn("600-800", model.prompt)
        self.assertIn("10-12 distinct", model.prompt)
        self.assertIn("https://example.org/p", report)

    def test_grounding_repair_rejects_collapsed_report(self):
        urls = [f"https://example.org/{index}" for index in range(8)]
        draft = " ".join(["Grounded factual sentence " * 18 + url for url in urls])
        collapsed = "Brief conclusion without sources."
        self.assertFalse(_repair_output_is_usable(collapsed, draft, "650-800", "8-10"))

        class Model:
            def generate(self, messages, **kwargs):
                return collapsed

        papers = [Paper(str(index), f"Paper {index}", 2020, url, "Direct evidence.") for index, url in enumerate(urls)]
        self.assertEqual(
            repair_grounded_report(draft, papers, Model(), "test-collapse", "650-800", "8-10"),
            draft,
        )

    def test_adaptive_search_uses_three_initial_and_two_feedback_queries(self):
        class PlannerSettings:
            model = "planner"

        class Planner:
            settings = PlannerSettings()

            def generate_json(self, messages, **kwargs):
                if "final two searches" in messages[0]["content"]:
                    return {"queries": ["Exact Feedback Paper One", "Exact Feedback Paper Two"]}
                return {"queries": [
                    "initial topic one", "initial topic two", "initial topic three",
                    "unused topic four", "unused topic five",
                ]}

        class Scholar:
            def search(self, query, **kwargs):
                return [Paper(query, f"Paper for {query}", 2020, f"https://example.org/{query}", "useful evidence")]

        class SearchSettings:
            baseline_search_budget = 5
            search_results_per_query = 5
            search_workers = 5

        task = load_tasks(Path("data/subsets/reportbench_30.jsonl"))[0]
        queries, papers = adaptive_search(task, Planner(), Scholar(), SearchSettings(), None)
        self.assertEqual(len(queries), 5)
        self.assertEqual(len(papers), 5)
        self.assertEqual({paper.search_query_index for paper in papers}, set(range(5)))

    def test_anonymous_free_provider_supplements_central_query(self):
        class Web:
            def search(self, query, **kwargs):
                return [Paper(query, f"Paper {query}", 2020, f"https://example.org/{query}", "evidence")]

        class OpenAlexProvider:
            mailto = ""
            calls = []

            def search(self, query, **kwargs):
                self.calls.append(query)
                return [Paper("W1", "Structured Central Paper", 2020, "https://openalex.org/W1", "evidence")]

        supplement = OpenAlexProvider()
        papers = CompositeScholarProvider([Web(), supplement]).search_many(
            ["one query", "two query"], cutoff=None, limit=5, workers=2,
        )
        self.assertEqual(supplement.calls, ["one query"])
        self.assertEqual(len(papers), 3)

    def test_search_result_cleanup_rejects_wrappers_and_nonscholarly_pages(self):
        self.assertEqual(
            canonical_search_title("[1503.02531] Distilling the Knowledge in a Neural Network"),
            canonical_search_title("Distilling the Knowledge in a Neural Network"),
        )
        self.assertFalse(is_scholarly_candidate(Paper("x", "Client Challenge", 2020, "https://x.test", "text")))
        self.assertFalse(is_scholarly_candidate(Paper("x", "What is Knowledge Distillation?", 2020, "https://ibm.com/a", "text")))
        self.assertTrue(is_scholarly_candidate(Paper("x", "Distilling the Knowledge in a Neural Network", 2015, "https://arxiv.org/a", "text")))

    def test_minimax_search_uses_plain_query_and_local_cutoff(self):
        class CacheStub:
            def get_or_create(self, namespace, payload, factory):
                self.namespace = namespace
                self.payload = payload
                return {"organic": [
                    {"title": "Useful Older Paper", "link": "https://example.org/old", "date": "2020"},
                    {"title": "Future Paper", "link": "https://example.org/new", "date": "2026"},
                ], "base_resp": {"status_code": 0}}

        from datetime import date
        cache = CacheStub()
        provider = MiniMaxSearchProvider(cache, "unused", "https://example.org")
        papers = provider.search("knowledge distillation", cutoff=date(2021, 6, 1), limit=10)
        self.assertEqual(cache.namespace, "minimax-web-search-v2")
        self.assertNotIn("before:", cache.payload["q"])
        self.assertIn("academic paper", cache.payload["q"])
        self.assertEqual([paper.title for paper in papers], ["Useful Older Paper"])

    def test_page_reader_prefers_citation_metadata_and_extracts_body(self):
        parsed = parse_academic_html(
            '<html><head><title>Short - Site</title>'
            '<meta name="citation_title" content="A Complete Academic Paper Title">'
            '<meta name="citation_abstract" content="This abstract contains sufficiently detailed scholarly evidence for testing.">'
            '</head><body><nav><p>This navigation paragraph must be ignored completely.</p></nav>'
            '<p>This body paragraph contains additional experimental evidence and useful scientific details.</p>'
            '<script>not evidence</script></body></html>'
        )
        self.assertEqual(parsed["title"], "A Complete Academic Paper Title")
        self.assertIn("experimental evidence", parsed["text"])
        self.assertNotIn("navigation", parsed["text"])

    def test_arxiv_pdf_url_is_canonical_and_bounded(self):
        self.assertEqual(
            arxiv_pdf_url("https://export.arxiv.org/abs/2401.12345v2"),
            "https://arxiv.org/pdf/2401.12345",
        )
        self.assertEqual(arxiv_pdf_url("https://example.org/paper.pdf"), "")
        self.assertEqual(extract_pdf_text(b"not a pdf"), "")

    def test_page_reader_does_not_replace_complete_search_title(self):
        from reportbench_mm.web_reader import WebPageReader

        class Reader(WebPageReader):
            def read(self, url):
                return {"title": "Complete Paper Title | Publisher Website", "text": "body"}

        paper = Paper("p", "Complete Paper Title", 2020, "https://example.org")
        Reader(None).enrich(paper)
        self.assertEqual(paper.title, "Complete Paper Title")
        self.assertEqual(paper.full_text, "body")

    def test_page_reader_only_completes_matching_truncated_title(self):
        from reportbench_mm.web_reader import WebPageReader

        class GoodReader(WebPageReader):
            def read(self, url):
                return {"title": "Range-Doppler Detection in Automotive Radar with Deep Learning", "text": "body"}

        class ChallengeReader(WebPageReader):
            def read(self, url):
                return {"title": "Client Challenge", "text": ""}

        good = Paper("g", "Range-Doppler Detection in Automotive Radar with Deep ...", 2020, "https://x")
        bad = Paper("b", "Range-Doppler Detection in Automotive Radar with Deep ...", 2020, "https://x")
        GoodReader(None).enrich(good)
        ChallengeReader(None).enrich(bad)
        self.assertEqual(good.title, "Range-Doppler Detection in Automotive Radar with Deep Learning")
        self.assertIn("...", bad.title)

    def test_search_planner_rejects_generic_and_duplicate_queries(self):
        class PlannerSettings:
            model = "planner"

        class Planner:
            settings = PlannerSettings()

            def generate_json(self, messages, **kwargs):
                return {"queries": [
                    "person search deep metric learning before May 2021",
                    "person search deep metric learning",
                    "person search identity driven detection",
                    "bad",
                ]}

        task = load_tasks(Path("data/subsets/reportbench_30.jsonl"))[8]
        queries = plan_search_queries(task, Planner(), limit=3)
        self.assertEqual(len(queries), 3)
        self.assertNotIn("before", queries[0].lower())
        self.assertEqual(len({query.lower() for query in queries}), 3)

    def test_model_reranker_validates_indices_and_fills_missing_slots(self):
        class RerankerSettings:
            model = "reranker"

        class Reranker:
            settings = RerankerSettings()

            def generate_json(self, messages, **kwargs):
                return {"indices": [3, 3, 99, "2"]}

        task = load_tasks(Path("data/subsets/reportbench_30.jsonl"))[0]
        papers = [Paper(str(i), f"Paper {i}", 2020, "u", "abstract") for i in range(5)]
        selected = model_rerank_papers(task, papers, Reranker(), 4)
        self.assertEqual([paper.paper_id for paper in selected], ["2", "1", "0", "3"])

    def test_parallel_search_deduplicates_doi(self):
        class Scholar:
            def search(self, query, **kwargs):
                return [Paper(query, query, 2020, f"https://example.org/{query}", "abstract", doi="same")]

        papers = parallel_search(Scholar(), ["query one", "query two"], cutoff=None, workers=2)
        self.assertEqual(len(papers), 1)

    def test_diverse_selection_reserves_each_query(self):
        papers = []
        for query_index in range(3):
            for rank in range(3):
                paper = Paper(f"{query_index}-{rank}", f"Paper {query_index}-{rank}", 2020, "u", "a")
                paper.search_query_index = query_index
                paper.search_rank = rank
                paper.relevance = 1.0 if query_index == 0 else 0.1
                papers.append(paper)
        selected = diverse_top_papers(papers, query_count=3, limit=6)
        self.assertEqual({paper.search_query_index for paper in selected}, {0, 1, 2})

    def test_large_diverse_pool_caps_reserved_slots_per_query(self):
        papers = []
        for query_index in range(2):
            for rank in range(10):
                paper = Paper(f"{query_index}-{rank}", f"Paper {query_index}-{rank}", 2020, "u", "a")
                paper.search_query_index = query_index
                paper.search_rank = rank
                paper.relevance = 1.0 if query_index == 0 else 0.01
                papers.append(paper)
        selected = diverse_top_papers(papers, query_count=2, limit=10)
        self.assertEqual(sum(p.search_query_index == 1 for p in selected), 3)

    def test_invalid_embedded_json_is_a_recoverable_runtime_error(self):
        class BrokenJsonClient(MiniMaxClient):
            def generate(self, messages, **kwargs):
                return 'prefix {"decisions":[{"match":"unterminated}]}'

        client = BrokenJsonClient(object())
        with self.assertRaises(RuntimeError):
            client.generate_json([])

    def test_cache_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = JsonCache(Path(directory) / "cache.sqlite3")
            payload = {"b": 2, "a": 1}
            cache.put("test", payload, {"ok": True})
            self.assertEqual(cache.get("test", {"a": 1, "b": 2}), {"ok": True})

    def test_stratified_subset_is_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "tasks.jsonl"
            rows = [
                {"arxiv_id": str(i), "title": f"T{i}", "prompt": "P", "application_domain": f"D{i % 3}"}
                for i in range(12)
            ]
            source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            tasks = load_tasks(source)
            first = stratified_subset(tasks, 9, seed=4)
            second = stratified_subset(tasks, 9, seed=4)
            self.assertEqual([task.arxiv_id for task in first], [task.arxiv_id for task in second])
            self.assertEqual(len({task.application_domain for task in first[:3]}), 3)

    def test_cutoff_and_forbidden_filter(self):
        cutoff = extract_cutoff("Only cite papers published before April 2025.")
        self.assertEqual(cutoff.isoformat(), "2025-04-01")
        papers = [
            Paper("1", "The Forbidden Survey", 2024, "u1"),
            Paper("2", "Allowed Paper", 2024, "u2"),
            Paper("3", "Future Paper", 2026, "u3"),
        ]
        kept = filter_papers(papers, forbidden_title="The Forbidden Survey", cutoff=cutoff)
        self.assertEqual([paper.paper_id for paper in kept], ["2"])
        self.assertEqual(
            filter_papers(
                [Paper("4", "The Forbidden Survey ...", 2024, "u4")],
                forbidden_title="The Forbidden Survey With A Much Longer Subtitle", cutoff=cutoff,
            ),
            [],
        )
        self.assertEqual(
            filter_papers(
                [Paper("5", "[2310.12986] The Forbidden Survey With A Much ...", 2023, "u5")],
                forbidden_title="The Forbidden Survey With A Much Longer Subtitle", cutoff=cutoff,
            ),
            [],
        )
        self.assertEqual(normalize_title("A: Test!"), "a test")
        query = compact_query("Please help me research Knowledge Distillation and Student-Teacher Learning before June 2021")
        self.assertIn("Knowledge Distillation", query)
        self.assertNotIn("Please", query)
        queries = search_queries("I study Knowledge Distillation and Student-Teacher Learning, including logits-based distillation and self-distillation.")
        self.assertTrue(any("Knowledge Distillation" in query for query in queries))
        self.assertFalse(any(query.lower() == "single" for query in queries))
        watermark = search_queries("I study the application of deep learning in the field of image digital watermarking.")
        self.assertIn("image digital watermarking", watermark[0].lower())
        self.assertTrue(any("deep learning image digital watermarking" in query.lower() for query in watermark))
        speech = search_queries('Research the field of "artificial intelligence-based automated speech therapy tools applied to speech disorders."')
        self.assertIn("speech therapy", speech[0].lower())
        radar = search_queries("Research academic advancements in different radar data representation methods in the field of autonomous driving.")
        self.assertIn("radar data representation", radar[0].lower())
        self.assertIn("autonomous driving", radar[0].lower())

    def test_rag_scoring_prefers_relevant_paper(self):
        terms = keywords("graph neural networks for text classification")
        relevant = Paper("1", "Graph Neural Networks for Text Classification", 2023, "u", "Graph convolution for document labels", cited_by_count=20)
        unrelated = Paper("2", "Marine Biology Review", 2023, "u", "Fish populations and coral reefs", cited_by_count=1000)
        self.assertGreater(score_paper(relevant, terms), score_paper(unrelated, terms))
        self.assertEqual(anchor_coverage(relevant, {"graph", "neural", "networks"}), 1.0)
        self.assertEqual(anchor_coverage(unrelated, {"graph", "neural", "networks"}), 0.0)
        kd = Paper("3", "Knowledge Distillation for Vision", 2020, "u", "Teacher student compression")
        membrane = Paper("4", "Membrane Distillation", 2020, "u", "Knowledge about water purification")
        self.assertTrue(matches_anchor_phrase(kd, "Knowledge Distillation"))
        self.assertFalse(matches_anchor_phrase(membrane, "Knowledge Distillation"))

    def test_rag_writer_uses_smaller_evidence_budget(self):
        settings = Settings.load(Path.cwd())
        self.assertLess(settings.rag_evidence_papers, settings.rag_max_papers)
        self.assertGreaterEqual(settings.rag_output_tokens, 16384)
        self.assertLessEqual(settings.rag_output_tokens, settings.max_output_tokens)

    def test_writing_selection_keeps_deep_nodes_traversal_only(self):
        task = load_tasks(Path("data/subsets/reportbench_30.jsonl"))[0]
        direct = Paper("d", "Knowledge Distillation for Object Detection", 2020, "u1", "Teacher student object detection", cited_by_count=200, depth=0, relevance=0.4)
        canonical = Paper("c", "Distilling the Knowledge in a Neural Network", 2015, "u2", "Knowledge distillation teacher student", cited_by_count=10000, depth=1, relevance=0.3)
        deep = Paper("x", "Knowledge Distillation Background", 2010, "u3", "Knowledge distillation", cited_by_count=50000, depth=2, relevance=0.7)
        selected = select_writing_papers([deep, canonical, direct], task, 10)
        self.assertEqual({paper.paper_id for paper in selected}, {"d", "c"})
        self.assertGreater(writing_score(canonical, keywords(task.prompt), {"knowledge", "distillation"}), 0)
        weak_direct = Paper("w", "Knowledge Distillation Method", 2020, "u4", "Teacher student", cited_by_count=5, depth=0, relevance=canonical.relevance)
        self.assertGreater(
            writing_score(canonical, keywords(task.prompt), {"knowledge", "distillation"}),
            writing_score(weak_direct, keywords(task.prompt), {"knowledge", "distillation"}),
        )

    def test_writing_selection_reserves_direct_retrieval_evidence(self):
        task = load_tasks(Path("data/subsets/reportbench_30.jsonl"))[0]
        papers = []
        for index in range(16):
            paper = Paper(
                f"d{index}", f"Knowledge Distillation Direct Paper {index}", 2020, f"u{index}",
                "Knowledge distillation teacher student model compression", depth=0, relevance=0.25,
            )
            papers.append(paper)
        for index in range(16):
            paper = Paper(
                f"g{index}", f"Knowledge Distillation Graph Paper {index}", 2015, f"g{index}",
                "Knowledge distillation teacher student model compression", depth=1,
                relevance=0.6, cited_by_count=10000,
            )
            papers.append(paper)
        selected = select_writing_papers(papers, task, 16)
        self.assertGreaterEqual(sum(paper.depth == 0 for paper in selected), 11)

    def test_source_labels_are_normalized_to_urls(self):
        papers = [Paper("1", "Paper One", 2020, "https://example.org/one")]
        report = "A supported claim (Source 1).\n\n- Source 1: https://example.org/one"
        normalized = normalize_source_citations(report, papers)
        self.assertIn("(https://example.org/one)", normalized)
        self.assertIn("- https://example.org/one", normalized)
        self.assertNotIn("Source 1", normalized)

    def test_report_sanitizer_removes_multi_source_synthesis_and_bibliography(self):
        report = (
            "Atomic claim (https://example.org/a). Combined claim (https://example.org/a, https://example.org/b). "
            "Second atomic claim (https://example.org/b).\n\n"
            "**References**\n- https://example.org/a"
        )
        cleaned = sanitize_report(report)
        self.assertIn("Atomic claim", cleaned)
        self.assertIn("Second atomic claim", cleaned)
        self.assertNotIn("Combined claim", cleaned)
        self.assertNotIn("References", cleaned)

    def test_scholar_provider_falls_back_after_failure(self):
        class Failed:
            def search(self, *args, **kwargs):
                raise RuntimeError("limited")

        class FreeFallback:
            def search(self, *args, **kwargs):
                return [Paper("fallback", "Recovered Paper", 2020, "https://example.org")]

        provider = CompositeScholarProvider([Failed(), FreeFallback()])
        papers = provider.search("topic", cutoff=None, limit=3)
        self.assertEqual(papers[0].title, "Recovered Paper")

    def test_scholar_provider_supplements_abstractless_results(self):
        class Sparse:
            def search(self, *args, **kwargs):
                return [Paper("s", "Sparse Paper", 2020, "https://example.org/s")]

        class Rich:
            def search(self, *args, **kwargs):
                return [Paper("r", "Rich Paper", 2020, "https://example.org/r", "Useful abstract")]

        papers = CompositeScholarProvider([Sparse(), Rich()]).search("topic", cutoff=None, limit=3)
        self.assertEqual([paper.title for paper in papers], ["Sparse Paper", "Rich Paper"])

    def test_hybrid_search_merges_graph_metadata_into_web_hit(self):
        class Web:
            def search(self, query, **kwargs):
                return [Paper("MMSEARCH:1", "Canonical Method Paper", 2020, "https://search.test/p", "short", source="minimax-search")]

        class Structured:
            def search(self, query, **kwargs):
                return [Paper(
                    "W1", "Canonical Method Paper", 2020, "https://doi.org/10.1/test",
                    "A much longer structured abstract with usable evidence.", doi="10.1/test",
                    cited_by_count=100, referenced_work_ids=["W0"], source="openalex",
                )]

        papers = CompositeScholarProvider([Web(), Structured()]).search_many(
            ["central topic", "subtopic"], cutoff=None, limit=10, workers=2,
        )
        self.assertEqual(len(papers), 1)
        self.assertEqual(papers[0].paper_id, "W1")
        self.assertEqual(papers[0].referenced_work_ids, ["W0"])
        self.assertEqual(papers[0].query_hits, 2)
        self.assertIn("openalex", papers[0].source)

    def test_composite_batches_openalex_graph_nodes(self):
        class OpenAlexProvider:
            def get_works(self, paper_ids, depth=0):
                return [Paper(paper_id, f"Paper {paper_id}", 2020, "u", depth=depth) for paper_id in paper_ids]

            def get_work(self, paper_id, depth=0):
                return None

        papers = CompositeScholarProvider([OpenAlexProvider()]).get_works(["W1", "W2", "W1"], depth=2)
        self.assertEqual([paper.paper_id for paper in papers], ["W1", "W2"])
        self.assertTrue(all(paper.depth == 2 for paper in papers))

    def test_reference_matching_is_one_to_one(self):
        self.assertTrue(title_match("Graph Neural Networks: A Survey", "Graph Neural Networks A Survey"))
        self.assertEqual(maximum_matches(["Same Paper", "Same Paper"], ["Same Paper"]), 1)
        self.assertEqual(normalize_url("HTTPS://WWW.ARXIV.ORG/abs/1234/"), "https://arxiv.org/abs/1234")
        items = cited_statements("A factual claim (https://example.org/paper). Another sentence.")
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["url"], "https://example.org/paper")

    def test_noncited_extraction_splits_length_failures(self):
        class SettingsStub:
            judge_model = "judge"

        class SplittingModel:
            settings = SettingsStub()

            def generate_json(self, messages, **kwargs):
                candidates = json.loads(messages[0]["content"].split("CANDIDATES:\n", 1)[1])
                if len(candidates) > 1:
                    raise RuntimeError("MiniMax returned empty final content (finish_reason=length)")
                return {"statements": candidates}

        report = "\n".join([
            "A sufficiently long externally verifiable factual statement number one.",
            "A sufficiently long externally verifiable factual statement number two.",
        ])
        statements = extract_noncited(report, [], SplittingModel(), limit=20)
        self.assertEqual(len(statements), 2)

    def test_aggregate_reports_micro_and_nonempty_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                {"reference_matches": 1, "reference_count": 2, "ground_truth_count": 10,
                 "cited_supported": 3, "cited_count": 4, "noncited_correct": 0,
                 "noncited_count": 0, "noncited_factual_accuracy": 0},
                {"reference_matches": 2, "reference_count": 3, "ground_truth_count": 5,
                 "cited_supported": 1, "cited_count": 2, "noncited_correct": 2,
                 "noncited_count": 2, "noncited_factual_accuracy": 1},
            ]
            paths = []
            for index, row in enumerate(rows):
                path = root / f"{index}.json"
                path.write_text(json.dumps(row), encoding="utf-8")
                paths.append(path)
            summary = aggregate(paths, root / "summary.json")
            self.assertEqual(summary["reference_micro_precision"], 3 / 5)
            self.assertEqual(summary["reference_micro_recall"], 3 / 15)
            self.assertEqual(summary["noncited_micro_accuracy"], 1.0)
            self.assertEqual(summary["noncited_nonempty_task_count"], 1)


if __name__ == "__main__":
    unittest.main()
