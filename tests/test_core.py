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


class CoreTests(unittest.TestCase):
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
        self.assertGreater(settings.rag_output_tokens, settings.max_output_tokens)

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
