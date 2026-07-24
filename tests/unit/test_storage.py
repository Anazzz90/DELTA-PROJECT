"""
tests/unit/test_storage.py
============================
Checkpoint 9 -- Unit Tests for SQLite Storage + ChromaDB Vector Store

Coverage (maps exactly to checkpoint test criteria):
  Criterion 1 -- Full pipeline output saved to SQLite
  Criterion 2 -- SELECT * FROM queries returns correct row
  Criterion 3 -- Re-run same query -> second row added (history accumulates)
  Criterion 4 -- get_history() returns list of past results correctly
  Criterion 5 -- ChromaDB initialises cleanly (no errors)
  Criterion 6 -- VectorStore stores and retrieves a test embedding correctly

Uses an in-memory SQLite database so tests never write to disk.
ChromaDB uses a temp directory.

Run:
    poetry run pytest tests/unit/test_storage.py -v
"""

import asyncio
import tempfile
from pathlib import Path

import pytest
import pytest_asyncio

from core.aggregator import FinalDecision
from core.conflict_detector import ConflictReport
from core.schemas import AgentOutput, AgentResult
from core.scoring_engine import ScoringResult
from db.models import Base
from memory.vector_store import VectorStore


# =============================================================================
# In-memory DB fixture (no file written to disk during tests)
# =============================================================================

@pytest.fixture
async def db_session():
    """
    Create an in-memory async SQLite database, create all tables,
    and yield a HistoryStore pointed at it.
    """
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from db.session import AsyncSession

    # In-memory SQLite
    test_engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        future=True,
    )
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Patch the HistoryStore to use our in-memory engine
    from db import session as session_module
    original_engine           = session_module.engine
    original_session_local    = session_module.AsyncSessionLocal

    session_module.engine              = test_engine
    session_module.AsyncSessionLocal   = async_sessionmaker(
        bind=test_engine, class_=AsyncSession, expire_on_commit=False
    )

    from memory.history import HistoryStore
    store = HistoryStore()
    yield store

    # Restore
    session_module.engine            = original_engine
    session_module.AsyncSessionLocal = original_session_local
    await test_engine.dispose()


@pytest.fixture
def vector_store(tmp_path):
    """VectorStore pointed at a temp dir — deleted after each test."""
    return VectorStore(persist_dir=tmp_path / "chroma")


# =============================================================================
# Shared helpers
# =============================================================================

def make_agent_result(name: str, driver: str = "Volume surge", success: bool = True) -> AgentResult:
    output = AgentOutput(
        extracted_facts=["BTC volume up 3x"],
        possible_explanations=["Explanation A", "Explanation B"],
        ranked_hypotheses=["Hypothesis 1", "Hypothesis 2"],
        main_driver=driver,
        confidence_score=0.70,
        acknowledged_weaknesses=["Some uncertainty"],
    ) if success else None
    return AgentResult(
        agent_name=name,
        success=success,
        output=output,
        model="groq/llama-3.1-8b-instant",
        total_tokens=200,
        cost_usd=0.0001,
        latency_ms=400.0,
        error=None if success else "Simulated failure",
    )


def make_scoring_result(name: str, score: float = 0.75, overconfident: bool = False) -> ScoringResult:
    return ScoringResult(
        agent_name=name,
        fact_consistency_score=score,
        reasoning_depth_score=score,
        overconfidence_penalty=0.0,
        final_score=score,
        overconfident=overconfident,
    )


def make_final_decision() -> FinalDecision:
    return FinalDecision(
        system_main_driver="Liquidation cascade caused the spike",
        system_confidence_score=0.72,
        dominant_narratives=["liquidation_cascade", "volume_surge"],
        narrative_clusters={"liquidation_cascade": ["Hypothesis 1"]},
        contributing_agents=["neutral_analyst", "data_first"],
        conflict_adjusted=False,
        signal_summary={"positive": 2, "negative": 0},
        net_bias="Bullish",
        decision_logic="Majority of agents point to a liquidation cascade.",
    )


def make_conflict_report(detected: bool = False) -> ConflictReport:
    return ConflictReport(
        conflict_detected=detected,
        conflict_level="High Conflict" if detected else "Strong Consensus",
        conflict_type="none" if not detected else "narrative",
        conflicting_agents=[] if not detected else ["a", "b"],
        conflict_reason="No conflict" if not detected else "Opposing narratives",
    )


# =============================================================================
# Criterion 1+2 -- Outputs saved to SQLite, SELECT returns correct data
# =============================================================================

class TestSQLiteStorage:

    async def test_save_query_returns_integer_id(self, db_session):
        query_id = await db_session.save_query("Why did BTC spike?", ["Volume up 3x"])
        assert isinstance(query_id, int)
        assert query_id >= 1

    async def test_saved_query_retrievable_via_history(self, db_session):
        await db_session.save_query("Why did BTC spike?", ["Volume up 3x"])
        history = await db_session.get_history()
        assert len(history) == 1
        assert history[0]["question"] == "Why did BTC spike?"

    async def test_query_row_has_correct_fact_set(self, db_session):
        facts = ["BTC volume up 3x", "Derivatives liquidated"]
        await db_session.save_query("Test?", facts)
        history = await db_session.get_history()
        assert history[0]["fact_set"] == facts

    async def test_query_row_has_domain_profile(self, db_session):
        await db_session.save_query("Test?", ["fact"], domain_profile="intraday_trading")
        history = await db_session.get_history()
        assert history[0]["domain_profile"] == "intraday_trading"

    async def test_save_agent_output_no_error(self, db_session):
        query_id = await db_session.save_query("Test?", ["fact"])
        result   = make_agent_result("data_first", "Volume surge")
        scoring  = make_scoring_result("data_first", 0.75)
        # Should not raise
        await db_session.save_agent_output(query_id, result, scoring)

    async def test_save_failed_agent_output_no_error(self, db_session):
        query_id = await db_session.save_query("Test?", ["fact"])
        result   = make_agent_result("neutral_analyst", success=False)
        await db_session.save_agent_output(query_id, result)

    async def test_save_final_decision_no_error(self, db_session):
        query_id = await db_session.save_query("Test?", ["fact"])
        decision = make_final_decision()
        report   = make_conflict_report(detected=False)
        await db_session.save_final_decision(query_id, decision, report, total_cost_usd=0.0003)

    async def test_final_decision_appears_in_history(self, db_session):
        query_id = await db_session.save_query("Test?", ["fact"])
        decision = make_final_decision()
        report   = make_conflict_report()
        await db_session.save_final_decision(query_id, decision, report)
        history  = await db_session.get_history()
        assert history[0]["system_main_driver"] == "Liquidation cascade caused the spike"
        assert history[0]["system_confidence_score"] == 0.72


# =============================================================================
# Criterion 3 -- Re-run same query -> second row added (history accumulates)
# =============================================================================

class TestHistoryAccumulation:

    async def test_second_query_adds_second_row(self, db_session):
        await db_session.save_query("First query?", ["fact 1"])
        await db_session.save_query("Second query?", ["fact 2"])
        history = await db_session.get_history()
        assert len(history) == 2

    async def test_same_question_twice_creates_two_rows(self, db_session):
        q = "Why did BTC spike?"
        facts = ["Volume up", "Derivatives"]
        await db_session.save_query(q, facts)
        await db_session.save_query(q, facts)
        history = await db_session.get_history()
        assert len(history) == 2

    async def test_history_ordered_newest_first(self, db_session):
        await db_session.save_query("First", ["f1"])
        await db_session.save_query("Second", ["f2"])
        history = await db_session.get_history()
        assert history[0]["question"] == "Second"
        assert history[1]["question"] == "First"


# =============================================================================
# Criterion 4 -- get_history() returns list correctly
# =============================================================================

class TestGetHistory:

    async def test_get_history_returns_list(self, db_session):
        history = await db_session.get_history()
        assert isinstance(history, list)

    async def test_get_history_empty_when_no_queries(self, db_session):
        history = await db_session.get_history()
        assert history == []

    async def test_get_history_respects_limit(self, db_session):
        for i in range(5):
            await db_session.save_query(f"Query {i}", ["fact"])
        history = await db_session.get_history(limit=3)
        assert len(history) == 3

    async def test_get_query_detail_returns_agent_outputs(self, db_session):
        query_id = await db_session.save_query("Test?", ["fact"])
        await db_session.save_agent_output(
            query_id,
            make_agent_result("data_first"),
            make_scoring_result("data_first"),
        )
        detail = await db_session.get_query_detail(query_id)
        assert detail is not None
        assert len(detail["agent_outputs"]) == 1
        assert detail["agent_outputs"][0]["agent_name"] == "data_first"

    async def test_get_query_detail_returns_none_for_unknown_id(self, db_session):
        detail = await db_session.get_query_detail(99999)
        assert detail is None


# =============================================================================
# Criterion 5 -- ChromaDB initialises cleanly
# =============================================================================

class TestVectorStoreInit:

    def test_vector_store_initialises_without_error(self, vector_store):
        assert vector_store is not None

    def test_vector_store_starts_empty(self, vector_store):
        assert vector_store.count() == 0

    def test_search_on_empty_store_returns_empty_list(self, vector_store):
        results = vector_store.search("BTC spike")
        assert results == []


# =============================================================================
# Criterion 6 -- VectorStore stores and retrieves embeddings correctly
# =============================================================================

class TestVectorStoreOperations:

    def test_add_stores_one_document(self, vector_store):
        vector_store.add(query_id=1, question="Why did BTC spike?")
        assert vector_store.count() == 1

    def test_search_returns_stored_document(self, vector_store):
        vector_store.add(query_id=1, question="Why did BTC spike 8% in the last hour?")
        results = vector_store.search("BTC price surge", n_results=1)
        assert len(results) == 1
        assert results[0]["query_id"] == 1

    def test_search_returns_most_similar_first(self, vector_store):
        vector_store.add(query_id=1, question="Why did BTC spike?")
        vector_store.add(query_id=2, question="Why did gold prices drop?")
        results = vector_store.search("BTC cryptocurrency price increase", n_results=2)
        assert len(results) >= 1
        # The BTC-related query should rank higher (lower distance score)
        assert results[0]["query_id"] == 1

    def test_upsert_same_id_does_not_duplicate(self, vector_store):
        vector_store.add(query_id=1, question="Why did BTC spike?")
        vector_store.add(query_id=1, question="Why did BTC spike again?")
        assert vector_store.count() == 1

    def test_metadata_stored_and_retrieved(self, vector_store):
        vector_store.add(
            query_id=42,
            question="Why did ETH drop?",
            metadata={"domain": "intraday_trading", "confidence": 0.72},
        )
        results = vector_store.search("ETH price drop", n_results=1)
        assert results[0]["metadata"]["domain"] == "intraday_trading"
        assert results[0]["metadata"]["query_id"] == 42

    def test_distance_is_float_between_0_and_2(self, vector_store):
        vector_store.add(query_id=1, question="BTC spike analysis")
        results = vector_store.search("BTC spike", n_results=1)
        assert 0.0 <= results[0]["distance"] <= 2.0

    def test_clear_resets_store(self, vector_store):
        vector_store.add(query_id=1, question="Test")
        vector_store.clear()
        assert vector_store.count() == 0


# =============================================================================
# Checkpoint 21 — Agent Performance Stats
# =============================================================================

class TestAgentPerformanceStats:

    async def test_no_history_returns_empty_dict(self, db_session):
        stats = await db_session.get_agent_performance_stats()
        assert stats == {}

    async def test_single_agent_default_zeroed_when_queried_directly(self, db_session):
        stats = await db_session.get_agent_performance_stats("never_run_agent")
        assert stats["queries_run"] == 0
        assert stats["accuracy_rate"] == 0.0

    async def test_stats_aggregate_across_multiple_queries(self, db_session):
        for i in range(3):
            query_id = await db_session.save_query(f"Query {i}?", ["fact"])
            result = make_agent_result("data_first")
            scoring = make_scoring_result("data_first", score=0.6 + i * 0.1)
            await db_session.save_agent_output(query_id, result, scoring)

        stats = await db_session.get_agent_performance_stats("data_first")

        assert stats["queries_run"] == 3
        assert stats["success_rate"] == 1.0
        assert stats["avg_final_score"] == pytest.approx((0.6 + 0.7 + 0.8) / 3)

    async def test_flagged_count_reflects_overconfident_runs(self, db_session):
        query_id = await db_session.save_query("Q1?", ["fact"])
        await db_session.save_agent_output(
            query_id, make_agent_result("contrarian"),
            make_scoring_result("contrarian", overconfident=True),
        )
        query_id = await db_session.save_query("Q2?", ["fact"])
        await db_session.save_agent_output(
            query_id, make_agent_result("contrarian"),
            make_scoring_result("contrarian", overconfident=False),
        )

        stats = await db_session.get_agent_performance_stats("contrarian")

        assert stats["flagged_count"] == 1
        assert stats["accuracy_rate"] == pytest.approx(0.5)

    async def test_failed_runs_excluded_from_score_averages_but_counted(self, db_session):
        query_id = await db_session.save_query("Q1?", ["fact"])
        await db_session.save_agent_output(query_id, make_agent_result("skeptic", success=False))
        query_id = await db_session.save_query("Q2?", ["fact"])
        await db_session.save_agent_output(
            query_id, make_agent_result("skeptic"), make_scoring_result("skeptic", score=0.8)
        )

        stats = await db_session.get_agent_performance_stats("skeptic")

        assert stats["queries_run"] == 2
        assert stats["success_rate"] == 0.5
        assert stats["avg_final_score"] == pytest.approx(0.8)

    async def test_stats_without_agent_name_returns_all_agents(self, db_session):
        query_id = await db_session.save_query("Q1?", ["fact"])
        await db_session.save_agent_output(
            query_id, make_agent_result("neutral_analyst"), make_scoring_result("neutral_analyst")
        )
        query_id = await db_session.save_query("Q2?", ["fact"])
        await db_session.save_agent_output(
            query_id, make_agent_result("skeptic"), make_scoring_result("skeptic")
        )

        stats = await db_session.get_agent_performance_stats()

        assert set(stats.keys()) == {"neutral_analyst", "skeptic"}
