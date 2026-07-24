"""
dashboard/streamlit_app.py
============================
Checkpoint 10 -- DMARS Streamlit MVP Dashboard

Full visual interface for the Delta-First Multi-AI Reasoning System.
Wraps the entire pipeline (agents -> scoring -> conflict -> aggregation -> storage)
into a clean, premium web UI.

Run:
    poetry run streamlit run dashboard/streamlit_app.py
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ensure the project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

# ── Page config (MUST be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="DMARS — Delta-First Reasoning",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Late imports (after sys.path is set) ─────────────────────────────────────
from agents.contrarian import ContrarianAgent
from agents.data_first import DataFirstAgent
from agents.intuition import IntuitionAgent
from agents.meta_ai import MetaAIAgent
from agents.neutral_analyst import NeutralAnalyst
from agents.skeptic import SkepticAgent
from core.aggregator import Aggregator
from core.conflict_detector import ConflictDetector
from core.pipeline import Pipeline
from core.scoring_engine import ScoringEngine
from db.session import create_all_tables
from memory.history import HistoryStore
from memory.vector_store import VectorStore
from core.research_engine import ResearchEngine

# =============================================================================
# Custom CSS — Premium Dark UI
# =============================================================================

st.markdown("""
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Background ── */
.stApp { background: #0d1117; }
section[data-testid="stSidebar"] { background: #161b22; border-right: 1px solid #30363d; }

/* ── Headings ── */
h1, h2, h3 { color: #e6edf3; }
p, label, .stText { color: #8b949e; }

/* ── Inputs ── */
textarea, input[type="text"] {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    color: #e6edf3 !important;
    border-radius: 8px !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #238636, #2ea043);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 2rem;
    font-weight: 600;
    font-size: 1rem;
    width: 100%;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2ea043, #3fb950);
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(46,160,67,0.4);
}

/* ── Agent Cards ── */
.agent-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1.2rem;
    margin-bottom: 1rem;
    transition: transform 0.2s;
}
.agent-card:hover { transform: translateY(-2px); }

/* ── Metric boxes ── */
.metric-box {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}

/* ── Conflict banner ── */
.conflict-banner {
    background: linear-gradient(135deg, #3d1a00, #5c2800);
    border: 1px solid #f85149;
    border-radius: 10px;
    padding: 1rem 1.5rem;
    color: #f85149;
    font-weight: 600;
    margin-bottom: 1rem;
}

/* ── Agreement banner ── */
.agree-banner {
    background: linear-gradient(135deg, #0d2d13, #1a4d2a);
    border: 1px solid #3fb950;
    border-radius: 10px;
    padding: 1rem 1.5rem;
    color: #3fb950;
    font-weight: 600;
    margin-bottom: 1rem;
}

/* ── Final decision block ── */
.final-block {
    background: linear-gradient(135deg, #1c2b3a, #162032);
    border: 1px solid #388bfd;
    border-radius: 12px;
    padding: 1.5rem;
    margin-top: 1rem;
}

/* ── History rows ── */
.history-row {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 0.8rem 1.2rem;
    margin-bottom: 0.5rem;
}

/* ── Driver text ── */
.driver-text {
    font-size: 1.05rem;
    font-weight: 500;
    color: #e6edf3;
}

/* ── Section divider ── */
hr { border-color: #30363d; }

/* ── Sidebar labels ── */
.sidebar-label { color: #8b949e; font-size: 0.85rem; margin-bottom: 0.2rem; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Agent config — name, color, emoji, description
# =============================================================================

AGENT_CONFIG = {
    "neutral_analyst": {
        "label":       "Neutral Analyst",
        "emoji":       "⚖️",
        "color":       "#388bfd",
        "border":      "#1f6feb",
        "description": "Balanced reasoning · DeepSeek-V3 (SiliconFlow)",
        "cls":         NeutralAnalyst,
    },
    "data_first": {
        "label":       "Data-First",
        "emoji":       "📊",
        "color":       "#3fb950",
        "border":      "#238636",
        "description": "Strictly fact-bound · Llama-3.1-8B (Groq)",
        "cls":         DataFirstAgent,
    },
    "skeptic": {
        "label":       "The Skeptic",
        "emoji":       "🔍",
        "color":       "#d29922",
        "border":      "#9e6a03",
        "description": "Adversarial reasoning · Llama-3.3-70B (Groq)",
        "cls":         SkepticAgent,
    },
    "contrarian": {
        "label":       "Contrarian",
        "emoji":       "⚡",
        "color":       "#f85149",
        "border":      "#b91c1c",
        "description": "Independent logic · GLM-4-9B (SiliconFlow)",
        "cls":         ContrarianAgent,
    },
    "intuition": {
        "label":       "Intuition",
        "emoji":       "🔮",
        "color":       "#a78bfa",
        "border":      "#7c3aed",
        "description": "Pattern recognition · Qwen-2.5-32B (SiliconFlow)",
        "cls":         IntuitionAgent,
    },
}


# =============================================================================
# Init helpers
# =============================================================================

@st.cache_resource
def get_engine() -> ScoringEngine:
    return ScoringEngine()

@st.cache_resource
def get_vector_store() -> VectorStore:
    return VectorStore()

def init_db():
    asyncio.run(create_all_tables())

@st.cache_resource
def _init_db_once():
    init_db()
    return True

_init_db_once()


# =============================================================================
# Pipeline runner (sync wrapper for async pipeline)
# =============================================================================




# =============================================================================
# Sidebar — Inputs
# =============================================================================

with st.sidebar:
    st.markdown("## 🧠 DMARS")
    st.markdown("<p style='color:#8b949e;font-size:0.85rem;margin-top:-0.5rem;'>Delta-First Multi-AI Reasoning</p>", unsafe_allow_html=True)
    st.divider()

    st.markdown("### 📝 Your Question")
    question = st.text_area(
        label="question",
        label_visibility="collapsed",
        placeholder="e.g. Why did BTC spike 8% in the last hour?",
        height=100,
        key="question_input",
    )

    if "facts_input_val" not in st.session_state:
        st.session_state.facts_input_val = ""

    st.markdown("### 📋 Verified Facts")
    st.markdown("<p class='sidebar-label'>One fact per line. Only facts you can verify.</p>", unsafe_allow_html=True)
    facts_raw = st.text_area(
        label="facts",
        label_visibility="collapsed",
        value=st.session_state.facts_input_val,
        placeholder="BTC volume up 3x in 60 minutes\nLarge derivatives positions liquidated\nNo major news in that window",
        height=160,
        key="facts_input_widget",
    )
    # Sync widget back to state
    st.session_state.facts_input_val = facts_raw

    st.markdown("### 🌐 Domain Profile")
    domain_profile = st.selectbox(
        label="domain",
        label_visibility="collapsed",
        options=["intraday_trading", "general", "macroeconomics", "startup", "manufacturing"],
        key="domain_select",
    )

    st.markdown("### 🤖 Active Agents")
    selected_agents = []
    for key, cfg in AGENT_CONFIG.items():
        checked = st.checkbox(
            f"{cfg['emoji']} {cfg['label']}",
            value=True,
            key=f"agent_{key}",
        )
        if checked:
            selected_agents.append(key)

    st.divider()
    st.markdown("### Meta-AI Synthesis")
    meta_ai_enabled = st.checkbox(
        "Enable Meta-AI (DeepSeek-V3 final verdict)",
        value=True,
        key="meta_ai_toggle",
    )
    st.divider()
    st.divider()
    st.markdown("### 🔍 Research Tool")
    r_mode = st.radio("Research Mode", ["URL", "Topic"], horizontal=True, key="r_mode_sidebar")
    
    if r_mode == "URL":
        research_url = st.text_input("URL", placeholder="https://...", key="research_url_sidebar")
        research_topic = None
    else:
        research_topic = st.text_input("Topic", placeholder="e.g. BTC price drivers", key="research_topic_sidebar")
        research_url = None

    if st.button("🌐 Research & Populate", use_container_width=True):
        if r_mode == "URL" and not research_url.strip():
            st.warning("Please enter a URL first.")
        elif r_mode == "Topic" and not research_topic.strip():
            st.warning("Please enter a topic first.")
        else:
            with st.spinner("Searching & Extracting..." if r_mode == "Topic" else "Scraping & Extracting..."):
                try:
                    engine = ResearchEngine(redis_conn=None)
                    if r_mode == "URL":
                        res = asyncio.run(engine.run_research(research_url))
                    else:
                        res = asyncio.run(engine.run_topic_research(research_topic))
                    
                    st.session_state.facts_input_val = "\n".join(res["extracted_facts"])
                    st.success(f"Extracted {len(res['extracted_facts'])} facts!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Research failed: {e}")

    st.divider()
    analyze_clicked = st.button("⚡ Analyze", use_container_width=True)


# =============================================================================
# Main content — Tabs
# =============================================================================

tab_analysis, tab_research, tab_history = st.tabs(["🔬 Analysis", "🌐 Research", "📚 History"])

# ── ANALYSIS TAB ──────────────────────────────────────────────────────────────
with tab_analysis:

    if not analyze_clicked:
        # Welcome screen
        st.markdown("""
        <div style='text-align:center; padding: 4rem 2rem;'>
            <div style='font-size:4rem; margin-bottom:1rem;'>🧠</div>
            <h1 style='color:#e6edf3; margin-bottom:0.5rem;'>DMARS</h1>
            <p style='color:#8b949e; font-size:1.1rem; margin-bottom:2rem;'>
                Delta-First Multi-AI Reasoning System
            </p>
            <p style='color:#6e7681; max-width:500px; margin: 0 auto;'>
                Enter your question and verified facts in the sidebar, then click
                <strong style='color:#3fb950;'>⚡ Analyze</strong> to see 
                multiple AI agents debate the evidence simultaneously.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Quick-fill sample buttons
        st.markdown("#### 📌 Try a sample query:")
        col1, col2 = st.columns(2)
        with col1:
            st.info("**BTC Spike** — Why did BTC spike 8% in the last hour?")
        with col2:
            st.info("**SaaS Churn** — Why did we lose 18% of users this month?")

    else:
        # ── Validation ────────────────────────────────────────────────────────
        if not question.strip():
            st.error("⚠️ Please enter a question in the sidebar.")
            st.stop()
        if not facts_raw.strip():
            st.error("⚠️ Please enter at least one verified fact.")
            st.stop()
        if not selected_agents:
            st.error("⚠️ Please select at least one agent.")
            st.stop()

        fact_set = [f.strip() for f in facts_raw.strip().splitlines() if f.strip()]

        # ── RUN ANALYSIS ──────────────────────────────────────────────────────
        async def run_analysis_flow():
            # 1. Run 5-agent pipeline
            agents_to_run = [AGENT_CONFIG[k]["cls"]() for k in selected_agents]
            pipeline = Pipeline(agents=agents_to_run)
            pipeline_result = await pipeline.run(question, fact_set, domain_profile)

            # 2. Score
            scoring_engine = get_engine()
            scoring_results = []
            for r in pipeline_result.successful_results:
                sr = scoring_engine.score(r.output, fact_set, r.agent_name)
                scoring_results.append(sr)

            # 3. Conflict & Aggregation
            conflict_detector = ConflictDetector()
            conflict_report = conflict_detector.detect(scoring_results, pipeline_result.results)
            aggregator = Aggregator()
            final_decision = aggregator.aggregate(scoring_results, pipeline_result.results, conflict_report)

            # 4. Meta-AI synthesis (optional)
            meta_result = None
            if meta_ai_enabled:
                loop = asyncio.get_event_loop()
                meta_agent = MetaAIAgent()
                meta_result = await loop.run_in_executor(
                    None,
                    lambda: meta_agent.synthesize(pipeline_result.results, question, fact_set, domain_profile)
                )

            # 5. Storage
            store = HistoryStore()
            query_id = await store.save_query(question, fact_set, domain_profile)
            for r in pipeline_result.results:
                sr = next((s for s in scoring_results if s.agent_name == r.agent_name), None)
                await store.save_agent_output(query_id, r, sr)
            await store.save_final_decision(
                query_id, final_decision, conflict_report,
                total_cost_usd=pipeline_result.total_cost_usd()
            )
            vs = get_vector_store()
            vs.add(query_id, question, metadata={"domain": domain_profile, "confidence": final_decision.system_confidence_score})

            return pipeline_result, scoring_results, conflict_report, final_decision, query_id, meta_result

        # ── Execution ─────────────────────────────────────────────────────────
        spinner_msg = f"⚡ Running {len(selected_agents)} agents" + (" + Meta-AI..." if meta_ai_enabled else "...")
        with st.spinner(spinner_msg):
            t_start = time.perf_counter()
            try:
                pipeline_result, scoring_results, conflict_report, final_decision, query_id, meta_result = asyncio.run(run_analysis_flow())
                elapsed_ms = (time.perf_counter() - t_start) * 1000
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                st.stop()

        # ── Header metrics ────────────────────────────────────────────────────
        st.markdown(f"### 🔬 Analysis: *{question[:70]}{'...' if len(question)>70 else ''}*")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("⚡ Pipeline Time",    f"{elapsed_ms:.0f}ms")
        m2.metric("✅ Agents Succeeded", f"{pipeline_result.agents_succeeded}/{len(selected_agents)}")
        m3.metric("💰 Total Cost",       f"${pipeline_result.total_cost_usd():.5f}")
        m4.metric("🎯 System Confidence",f"{final_decision.system_confidence_score:.0%}")

        st.divider()

        # ── Conflict / Agreement Banner ───────────────────────────────────────
        if conflict_report.conflict_detected:
            st.markdown(f"""
            <div class='conflict-banner'>
                ⚠️ CONFLICT DETECTED — Type: <strong>{conflict_report.conflict_type.upper()}</strong><br>
                <small>{conflict_report.conflict_reason}</small>
            </div>
            """, unsafe_allow_html=True)
        else:
            n_agree = pipeline_result.agents_succeeded
            total   = len(selected_agents)
            st.markdown(f"""
            <div class='agree-banner'>
                ✅ {n_agree}/{total} Agents in Agreement — No significant conflict detected
            </div>
            """, unsafe_allow_html=True)

        # ── Agent Response Cards ──────────────────────────────────────────────
        st.markdown("### 🤖 Agent Responses")
        cols = st.columns(len(pipeline_result.results))
        for col, result in zip(cols, pipeline_result.results):
            cfg = AGENT_CONFIG.get(result.agent_name, {})
            color   = cfg.get("color",   "#8b949e")
            border  = cfg.get("border",  "#30363d")
            emoji   = cfg.get("emoji",   "🤖")
            label   = cfg.get("label",   result.agent_name)
            desc    = cfg.get("description", "")

            with col:
                if result.success and result.output:
                    sr = next((s for s in scoring_results if s.agent_name == result.agent_name), None)
                    score_val = sr.final_score if sr else None
                    oc_flag   = sr.overconfident if sr else False

                    st.markdown(f"""
                    <div class='agent-card' style='border-color:{border};'>
                        <div style='color:{color}; font-weight:700; font-size:1.1rem;'>
                            {emoji} {label}
                        </div>
                        <div style='color:#6e7681; font-size:0.75rem; margin-bottom:0.8rem;'>{desc}</div>
                        <hr style='border-color:#30363d; margin:0.5rem 0;'>
                        <div style='color:#8b949e; font-size:0.75rem; margin-bottom:0.3rem;'>MAIN DRIVER</div>
                        <div class='driver-text'>{result.output.main_driver}</div>
                        <div style='margin-top:0.8rem; color:#8b949e; font-size:0.75rem;'>CONFIDENCE</div>
                    </div>
                    """, unsafe_allow_html=True)

                    safe_conf = min(1.0, max(0.0, float(result.output.confidence_score)))
                    st.progress(safe_conf,
                                text=f"{result.output.confidence_score:.0%}" +
                                (" ⚠️ overconfident" if oc_flag else ""))

                    if score_val is not None:
                        st.caption(f"Quality Score: **{score_val:.3f}**")

                    with st.expander("🔎 Full reasoning"):
                        st.markdown("**Extracted Facts:**")
                        for f in result.output.extracted_facts:
                            st.markdown(f"  - {f}")
                        st.markdown("**Ranked Hypotheses:**")
                        for h in result.output.ranked_hypotheses:
                            st.markdown(f"  - {h}")
                        st.markdown("**Acknowledged Weaknesses:**")
                        for w in result.output.acknowledged_weaknesses:
                            st.markdown(f"  - {w}")
                        st.caption(f"Model: `{result.model}` | Tokens: {result.total_tokens} | {result.latency_ms:.0f}ms")
                else:
                    st.markdown(f"""
                    <div class='agent-card' style='border-color:#f85149; opacity:0.7;'>
                        <div style='color:#f85149; font-weight:700;'>{emoji} {label}</div>
                        <div style='color:#8b949e; font-size:0.75rem;'>{desc}</div>
                        <hr style='border-color:#30363d;'>
                        <div style='color:#f85149;'>⚠️ Agent Failed</div>
                        <div style='color:#6e7681; font-size:0.8rem; margin-top:0.5rem;'>
                            {result.error[:120] if result.error else 'Unknown error'}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        # ── System Confidence Gauge ───────────────────────────────────────────
        st.divider()
        st.markdown("### 🎯 System Confidence")
        conf = final_decision.system_confidence_score
        conf_color = "#3fb950" if conf >= 0.65 else "#d29922" if conf >= 0.40 else "#f85149"
        st.markdown(f"""
        <div style='display:flex; align-items:center; gap:1rem; margin-bottom:0.5rem;'>
            <div style='font-size:2rem; font-weight:700; color:{conf_color};'>{conf:.0%}</div>
            <div style='color:#8b949e;'>{"HIGH confidence" if conf >= 0.65 else "MODERATE confidence" if conf>=0.40 else "LOW confidence"}
            {"— adjusted for conflict" if final_decision.conflict_adjusted else ""}</div>
        </div>
        """, unsafe_allow_html=True)
        safe_sys_conf = min(1.0, max(0.0, float(conf)))
        st.progress(safe_sys_conf)

        # ── Dominant Narratives ───────────────────────────────────────────────
        if final_decision.dominant_narratives:
            st.markdown("**Dominant Narrative Clusters:**")
            ncols = st.columns(len(final_decision.dominant_narratives[:3]))
            for nc, narr in zip(ncols, final_decision.dominant_narratives[:3]):
                count = len(final_decision.narrative_clusters.get(narr, []))
                nc.metric(narr.replace("_", " ").title(), f"{count} hypothesis votes")

        # ── Final Decision Block ──────────────────────────────────────────────
        st.markdown("### 🏁 Final System Decision")
        
        # Bias Color
        bias_color = "#3fb950" if final_decision.net_bias == "Bullish" else "#f85149" if final_decision.net_bias == "Bearish" else "#8b949e"
        
        st.markdown(f"""
        <div class='final-block'>
            <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:1rem;'>
                <div>
                    <div style='color:#8b949e; font-size:0.75rem;'>NET BIAS</div>
                    <div style='color:{bias_color}; font-size:1.2rem; font-weight:700;'>{final_decision.net_bias.upper()}</div>
                </div>
                <div style='text-align:right;'>
                    <div style='color:#8b949e; font-size:0.75rem;'>SIGNALS</div>
                    <div style='color:#e6edf3; font-size:1rem; font-weight:600;'>
                        <span style='color:#3fb950;'>+{final_decision.signal_summary['positive']}</span> / 
                        <span style='color:#f85149;'>-{final_decision.signal_summary['negative']}</span> / 
                        <span style='color:#8b949e;'>{final_decision.signal_summary['neutral']}</span>
                    </div>
                </div>
            </div>
            
            <div style='color:#8b949e; font-size:0.8rem; margin-bottom:0.4rem;'>FINAL DECISION</div>
            <div style='color:#e6edf3; font-size:1.15rem; font-weight:600; margin-bottom:1rem; line-height:1.4;'>
                {final_decision.system_main_driver}
            </div>

            <div style='background:rgba(0,0,0,0.2); border-radius:8px; padding:1rem; border-left:3px solid #388bfd;'>
                <div style='color:#8b949e; font-size:0.75rem; margin-bottom:0.3rem;'>DECISION LOGIC</div>
                <div style='color:#c9d1d9; font-size:0.9rem; line-height:1.5;'>{final_decision.decision_logic}</div>
            </div>
            
            <div style='color:#8b949e; font-size:0.75rem; margin-top:1.2rem;'>
                Contributing: {' · '.join(final_decision.contributing_agents)}
                &nbsp;|&nbsp; Conflict: <strong>{conflict_report.conflict_level}</strong>
                &nbsp;|&nbsp; Query ID: #{query_id}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Meta-AI Verdict Block (Checkpoint 12) ─────────────────────────────
        if meta_ai_enabled and meta_result:
            st.divider()
            st.markdown("### Meta-AI Final Verdict")
            if meta_result.success and meta_result.output:
                mo = meta_result.output
                conf_c = "#3fb950" if mo.final_confidence >= 0.65 else "#d29922" if mo.final_confidence >= 0.40 else "#f85149"
                st.markdown(f"""
                <div style='background:linear-gradient(135deg,#1a1f35,#0d1117); border:1px solid #388bfd;
                            border-radius:14px; padding:1.5rem; margin-top:0.5rem;'>
                    <div style='display:flex; align-items:center; gap:0.8rem; margin-bottom:1rem;'>
                        <div style='font-size:1.5rem;'>Meta-AI</div>
                        <div>
                            <div style='color:#388bfd; font-weight:700; font-size:1rem;'>DeepSeek-V3 (SiliconFlow)</div>
                            <div style='color:#6e7681; font-size:0.75rem;'>Final Synthesis Layer — Checkpoint 12</div>
                        </div>
                        <div style='margin-left:auto; font-size:1.6rem; font-weight:800; color:{conf_c};'>{mo.final_confidence:.0%}</div>
                    </div>
                    <div style='color:#8b949e; font-size:0.75rem; margin-bottom:0.3rem;'>DOMINANT DRIVER</div>
                    <div style='color:#e6edf3; font-size:1.05rem; font-weight:600; margin-bottom:1rem;'>{mo.dominant_driver}</div>
                    <div style='color:#8b949e; font-size:0.75rem; margin-bottom:0.3rem;'>SYNTHESIS</div>
                    <div style='color:#c9d1d9; font-size:0.9rem; line-height:1.6; margin-bottom:1rem;'>{mo.synthesis_conclusion}</div>
                    <div style='color:#8b949e; font-size:0.75rem; margin-bottom:0.3rem;'>RECOMMENDED ACTION</div>
                    <div style='color:#3fb950; font-size:0.9rem; font-weight:500;'>{mo.recommended_action}</div>
                    <div style='margin-top:1rem; color:#6e7681; font-size:0.75rem;'>
                        Supporting: {', '.join(mo.supporting_agents)}
                        &nbsp;|&nbsp; Model: <code>deepseek-ai/DeepSeek-V3</code>
                        &nbsp;|&nbsp; {meta_result.total_tokens} tokens | {meta_result.latency_ms:.0f}ms
                    </div>
                </div>
                """, unsafe_allow_html=True)
                if mo.minority_views:
                    with st.expander("Minority views preserved by Meta-AI"):
                        for mv in mo.minority_views:
                            st.markdown(f"- {mv}")
            else:
                st.warning(f"Meta-AI synthesis failed: {meta_result.error}")


# ── RESEARCH TAB ──────────────────────────────────────────────────────────────
with tab_research:
    st.markdown("### 🌐 Firecrawl Automated Research")
    st.markdown("<p style='color:#8b949e;'>Enter a URL or a Topic to perform a high-fidelity truth-filtering research run.</p>", unsafe_allow_html=True)
    
    r_tab_mode = st.radio("Research Mode", ["URL", "Topic"], horizontal=True, key="r_tab_mode")
    
    if r_tab_mode == "URL":
        r_url = st.text_input("Target URL", placeholder="https://...", key="research_tab_url")
        r_topic = None
    else:
        r_topic = st.text_input("Search Topic", placeholder="e.g. Lithium supply chain risks 2024", key="research_tab_topic")
        r_url = None

    r_domain = st.selectbox("Research Context", ["general", "crypto", "politics", "tech", "finance"], key="research_tab_domain")
    
    if st.button("🚀 Run Deep Research", use_container_width=True):
        if r_tab_mode == "URL" and not r_url.strip():
            st.error("Please enter a URL.")
        elif r_tab_mode == "Topic" and not r_topic.strip():
            st.error("Please enter a topic.")
        else:
            with st.spinner("Searching, Crawling, and Filtering..." if r_tab_mode == "Topic" else "Crawling, Extracting, and Filtering..."):
                try:
                    engine = ResearchEngine(redis_conn=None)
                    if r_tab_mode == "URL":
                        res = asyncio.run(engine.run_research(r_url, domain_profile=r_domain))
                    else:
                        res = asyncio.run(engine.run_topic_research(r_topic, domain_profile=r_domain))
                    
                    st.balloons()
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Facts Found", len(res["extracted_facts"]))
                    c2.metric("Conflict Score", f"{res['conflict_score']:.2f}")
                    c3.metric("Cache Status", res["cache_status"].upper())
                    
                    st.divider()
                    
                    col_left, col_right = st.columns([2, 1])
                    
                    with col_left:
                        st.markdown("#### 📋 Extracted & Validated Facts")
                        for fact in res["confidence_tiers"]:
                            tier_color = "#3fb950" if fact["tier"] == "verified" else "#d29922" if fact["tier"] == "inferred" else "#f85149"
                            st.markdown(f"""
                            <div style='background:#161b22; border-left:4px solid {tier_color}; padding:10px; border-radius:4px; margin-bottom:8px; font-size:0.9rem;'>
                                <span style='color:#8b949e; font-size:0.7rem;'>{fact['tier'].upper()}</span><br>
                                <span style='color:#e6edf3;'>{fact['fact']}</span>
                            </div>
                            """, unsafe_allow_html=True)
                            
                    with col_right:
                        st.markdown("#### 🔍 Metadata")
                        st.json(res["crawl_metadata"])
                        
                        if res["contradictions"]:
                            st.markdown("#### ⚠️ Contradictions")
                            for con in res["contradictions"]:
                                st.warning(con)
                                
                    if st.button("📥 Use these facts for Analysis"):
                        st.session_state.facts_input_val = "\n".join(res["extracted_facts"])
                        st.success("Facts moved to Analysis sidebar!")
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"Research failed: {e}")

# ── HISTORY TAB ───────────────────────────────────────────────────────────────
with tab_history:
    st.markdown("### 📚 Query History")

    history_store = HistoryStore()
    history = asyncio.run(history_store.get_history(limit=50))

    if not history:
        st.markdown("""
        <div style='text-align:center; padding:3rem; color:#6e7681;'>
            <div style='font-size:2rem;'>📭</div>
            <p>No queries yet. Run your first analysis!</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.caption(f"{len(history)} past queries")
        for item in history:
            conf     = item.get("system_confidence_score")
            conflict = item.get("conflict_detected")
            driver   = item.get("system_main_driver") or "—"
            question = item.get("question") or "—"
            created  = (item.get("created_at") or "")[:19].replace("T", " ")

            conf_str    = f"{conf:.0%}" if conf is not None else "—"
            conf_color  = "#3fb950" if (conf or 0) >= 0.65 else "#d29922" if (conf or 0) >= 0.40 else "#f85149"
            flag        = "⚠️ Conflict" if conflict else "✅ Agreement"

            st.markdown(f"""
            <div class='history-row'>
                <div style='display:flex; justify-content:space-between; align-items:flex-start;'>
                    <div>
                        <div style='color:#e6edf3; font-weight:500;'>#{item["id"]} — {question[:80]}</div>
                        <div style='color:#6e7681; font-size:0.8rem; margin-top:0.2rem;'>{driver[:90]}</div>
                    </div>
                    <div style='text-align:right; flex-shrink:0; margin-left:1rem;'>
                        <div style='color:{conf_color}; font-weight:700;'>{conf_str}</div>
                        <div style='color:#6e7681; font-size:0.75rem;'>{flag}</div>
                        <div style='color:#444d56; font-size:0.7rem;'>{created}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        if st.button("🗑️ Clear History Display", key="clear_hist"):
            st.info("Note: This only clears the display. Data is still in SQLite.")
