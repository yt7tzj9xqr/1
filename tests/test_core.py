import json
from pathlib import Path
import tempfile
import unittest

from reportbench_mm.cache import JsonCache
from reportbench_mm.dataset import load_tasks, stratified_subset
from reportbench_mm.providers.openalex import compact_query, extract_cutoff, filter_papers, normalize_title, search_queries
from reportbench_mm.schemas import Paper
from reportbench_mm.pipelines.rag import anchor_coverage, keywords, matches_anchor_phrase, score_paper
from reportbench_mm.config import Settings
from reportbench_mm.evaluation.reference import maximum_matches, normalize_url, title_match
from reportbench_mm.evaluation.statements import cited_statements


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

    def test_reference_matching_is_one_to_one(self):
        self.assertTrue(title_match("Graph Neural Networks: A Survey", "Graph Neural Networks A Survey"))
        self.assertEqual(maximum_matches(["Same Paper", "Same Paper"], ["Same Paper"]), 1)
        self.assertEqual(normalize_url("HTTPS://WWW.ARXIV.ORG/abs/1234/"), "https://arxiv.org/abs/1234")
        items = cited_statements("A factual claim (https://example.org/paper). Another sentence.")
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["url"], "https://example.org/paper")


if __name__ == "__main__":
    unittest.main()
