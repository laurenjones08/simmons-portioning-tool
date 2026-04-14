"""Home view for single-page router."""
import streamlit as st
from datetime import datetime


def _status_badge(status: str) -> str:
    colors = {
        "completed": ("#4CAF50", "#e8f5e9"),
        "running": ("#0046AD", "#e3f0ff"),
        "pending": ("#FFB74D", "#fff8e1"),
        "failed": ("#D9534F", "#fdecea"),
        "cancelled": ("#6b7280", "#f3f4f6"),
    }
    fg, bg = colors.get(status, ("#333", "#eee"))
    return (
        f"<span style='background:{bg};color:{fg};padding:2px 8px;"
        f"border-radius:12px;font-size:12px;font-weight:600'>{status.upper()}</span>"
    )


def render():
    # ── DESCRIPTION ───────────────────────────────────────────────────────
    st.markdown(
        "<div style='padding:0 4px;margin-bottom:18px'>"
        "<p style='font-size:15px;color:#333;line-height:1.7;margin:0'>"
        "This platform provides data-driven decision support for poultry portioning and production scheduling "
        "at Simmons Prepared Foods. Raw chicken breast (WIP) must be portioned into customer-specific products — "
        "fillets, tenders, nuggets/strips, and trim. Every portioning decision directly impacts <strong>Upgrade</strong>, "
        "the percentage of raw breast converted into sellable product, which is the plant's primary operational KPI."
        "<br/><br/>"
        "The platform digitizes and optimizes this decision process through two integrated models: an "
        "<strong>Enumeration Engine</strong> that scores every feasible cut combination, and a "
        "<strong>Scheduling Optimizer</strong> that translates the best combinations into a production schedule "
        "aligned with customer demand, line capacity, and labor constraints."
        "</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── PLATFORM MODULES ──────────────────────────────────────────────────
    st.subheader("Platform Modules")
    ov1, ov2, ov3 = st.columns(3)
    ov1.markdown(
        "<div class='simmons-card' style='min-height:160px'>"
        "<div style='font-size:24px;margin-bottom:6px'>🧭</div>"
        "<strong style='font-size:15px'>Enumeration Engine</strong>"
        "<div class='simmons-small' style='margin-top:8px'>"
        "Evaluates every feasible portioning combination against bird-size buckets, "
        "customer rules, plant constraints, and trim thresholds. "
        "Ranks results by upgrade %, economic value, and trim % to produce a "
        "ranked master list used as scheduling input."
        "</div></div>",
        unsafe_allow_html=True,
    )
    ov2.markdown(
        "<div class='simmons-card' style='min-height:160px'>"
        "<div style='font-size:24px;margin-bottom:6px'>📅</div>"
        "<strong style='font-size:15px'>Scheduling Optimizer</strong>"
        "<div class='simmons-small' style='margin-top:8px'>"
        "Ingests enumeration snapshots and solves a production schedule across a "
        "configurable planning horizon. Incorporates customer demand, inventory levels, "
        "line capacity, belt speed, and shift constraints to produce short- and long-term "
        "production recommendations."
        "</div></div>",
        unsafe_allow_html=True,
    )
    ov3.markdown(
        "<div class='simmons-card' style='min-height:160px'>"
        "<div style='font-size:24px;margin-bottom:6px'>📊</div>"
        "<strong style='font-size:15px'>Analytics &amp; Reporting</strong>"
        "<div class='simmons-small' style='margin-top:8px'>"
        "Surfaces upgrade %, trim %, value metrics, and job history across enumeration "
        "and scheduling runs. Provides exportable snapshots, scheduling decisions, and "
        "production output records for operators, planners, and management."
        "</div></div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── RECENT ACTIVITY ───────────────────────────────────────────────────
    st.subheader("Recent Activity")
    ra1, ra2, ra3 = st.columns(3)

    with ra1:
        last_enum_run = "—"
        last_enum_status = "—"
        last_enum_id = "—"
        last_enum_ts = "—"
        try:
            from api_client import list_jobs
            ejobs = list_jobs() or []
            if ejobs:
                e = sorted(ejobs, key=lambda j: j.get("createdAt", ""), reverse=True)[0]
                last_enum_run = e.get("runId", "—")
                last_enum_status = e.get("status", "—")
                raw_id = e.get("jobId") or e.get("_id", "—")
                last_enum_id = raw_id[:12] + "..." if len(raw_id) > 12 else raw_id
                ts = e.get("createdAt", "")
                if ts:
                    try:
                        last_enum_ts = datetime.fromisoformat(ts.replace("Z", "+00:00")).strftime("%b %d %H:%M UTC")
                    except Exception:
                        last_enum_ts = ts
        except Exception:
            pass
        badge = _status_badge(last_enum_status) if last_enum_status != "—" else "<span class='simmons-small'>—</span>"
        st.markdown(
            f"<div class='simmons-card'>"
            f"<div style='font-weight:700;margin-bottom:6px'>Last Enumeration Run</div>"
            f"<div class='simmons-small'>Run ID: <strong>{last_enum_run}</strong></div>"
            f"<div class='simmons-small'>Job ID: <code>{last_enum_id}</code></div>"
            f"<div class='simmons-small'>Submitted: {last_enum_ts}</div>"
            f"<div style='margin-top:8px'>Status: {badge}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with ra2:
        last_sched_run = "—"
        last_sched_status = "—"
        last_sched_id = "—"
        last_sched_ts = "—"
        try:
            from api_client import list_scheduling_jobs
            sjobs = list_scheduling_jobs() or []
            if sjobs:
                s = sorted(sjobs, key=lambda j: j.get("createdAt", ""), reverse=True)[0]
                last_sched_run = s.get("runId", "—")
                last_sched_status = s.get("status", "—")
                raw_sid = s.get("jobId") or s.get("_id", "—")
                last_sched_id = raw_sid[:12] + "..." if len(raw_sid) > 12 else raw_sid
                ts = s.get("createdAt", "")
                if ts:
                    try:
                        last_sched_ts = datetime.fromisoformat(ts.replace("Z", "+00:00")).strftime("%b %d %H:%M UTC")
                    except Exception:
                        last_sched_ts = ts
        except Exception:
            pass
        badge_s = _status_badge(last_sched_status) if last_sched_status != "—" else "<span class='simmons-small'>—</span>"
        st.markdown(
            f"<div class='simmons-card'>"
            f"<div style='font-weight:700;margin-bottom:6px'>Last Scheduling Run</div>"
            f"<div class='simmons-small'>Run ID: <strong>{last_sched_run}</strong></div>"
            f"<div class='simmons-small'>Job ID: <code>{last_sched_id}</code></div>"
            f"<div class='simmons-small'>Submitted: {last_sched_ts}</div>"
            f"<div style='margin-top:8px'>Status: {badge_s}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with ra3:
        snapshot_count = 0
        try:
            from api_client import search_mixes
            mixes = search_mixes({}) or []
            snapshot_count = len(mixes)
        except Exception:
            pass
        st.markdown(
            f"<div class='simmons-card'>"
            f"<div style='font-weight:700;margin-bottom:6px'>Enumeration Snapshots</div>"
            f"<div style='font-size:36px;font-weight:800;color:#0046AD;line-height:1.1'>{snapshot_count}</div>"
            f"<div class='simmons-small' style='margin-top:4px'>snapshots stored in MongoDB</div>"
            f"<div class='simmons-small'>available as scheduling inputs</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── DOCUMENTATION & RESOURCES ─────────────────────────────────────────
    st.subheader("Documentation & Resources")
    docs = [
        ("📖", "User Documentation", "Platform user guide and workflow manual.", "http://localhost:3000"),
        ("🔗", "Git Repository", "Source code, CI pipelines, and version history.", "https://github.com/laurenjones08/simmons-portioning-tool"),
        ("🚀", "Database Quick Start", "MongoDB bootstrap and common DB operations.", "https://github.com/laurenjones08/simmons-portioning-tool/blob/main/QUICK_START_DB.md"),
        ("🧭", "API Gateway", "Local service endpoints and Swagger API docs.", "http://localhost:8080"),
        ("🐳", "Docker Setup", "Container deployment and environment setup.", "https://github.com/laurenjones08/simmons-portioning-tool/blob/main/README.md"),
        ("🔧", "Troubleshooting", "Common issues, error codes, and fixes.", "https://github.com/laurenjones08/simmons-portioning-tool/blob/main/MICROSERVICE_API_ONBOARDING.md"),
    ]
    for row_start in range(0, len(docs), 3):
        doc_cols = st.columns(3)
        for i, col in enumerate(doc_cols):
            idx = row_start + i
            if idx < len(docs):
                icon, title, desc, link = docs[idx]
                col.markdown(
                    f"<div class='simmons-card' style='min-height:120px'>"
                    f"<div style='font-size:18px'>{icon} <strong>{title}</strong></div>"
                    f"<div class='simmons-small' style='margin-top:6px'>{desc}</div>"
                    f"<div style='margin-top:10px'>"
                    f"<a href='{link}' target='_blank'>"
                    f"<button style='padding:6px 14px;border-radius:6px;background:#0046AD;color:#fff;border:none;cursor:pointer;font-size:13px'>Open</button>"
                    f"</a></div></div>",
                    unsafe_allow_html=True,
                )

    st.markdown("---")

    # ── SYSTEM ARCHITECTURE ───────────────────────────────────────────────
    st.subheader("System Architecture")
    st.markdown(
        "<div class='simmons-card'>"
        "<div style='display:flex;gap:28px;align-items:flex-start;flex-wrap:wrap'>"
        "<div style='flex:1;min-width:220px'>"
        "<div style='font-weight:700;margin-bottom:6px'>🐳 Docker Microservices</div>"
        "<div class='simmons-small'>"
        "All backend services run in isolated Docker containers orchestrated by "
        "<code>docker-compose</code>. Services communicate over an internal Docker network; "
        "the Nginx API Gateway exposes a unified entry point at <code>localhost:8080</code>."
        "</div></div>"
        "<div style='flex:1;min-width:220px'>"
        "<div style='font-weight:700;margin-bottom:6px'>⚙ API Layer</div>"
        "<div class='simmons-small'>"
        "· <strong>Enumeration API</strong> — SKUs, buckets, cut strategies, mixes<br/>"
        "· <strong>Enumeration Worker</strong> — background enumeration jobs<br/>"
        "· <strong>Scheduling API</strong> — decisions, outputs, SKU demand<br/>"
        "· <strong>Scheduling Worker</strong> — background scheduling jobs<br/>"
        "· <strong>Config API</strong> — global config, production lines"
        "</div></div>"
        "<div style='flex:1;min-width:220px'>"
        "<div style='font-weight:700;margin-bottom:6px'>🗄 MongoDB Persistence</div>"
        "<div class='simmons-small'>"
        "Enumeration snapshots, scheduling decisions, produced outputs, and job metadata are "
        "stored as immutable versioned records. This supports auditing, historical comparison, "
        "and snapshot replay across runs."
        "</div></div>"
        "</div></div>",
        unsafe_allow_html=True,
    )

    st.caption("Simmons Portioning Optimization Platform — for schedulers, planners, operators, and management.")