"""Scheduling Dashboard view for single-page router."""
import streamlit as st
import pandas as pd
from datetime import datetime, timezone, timedelta
from api_client import (
    APIError,
    search_mixes,
    search_scheduling_decisions,
    search_scheduling_outputs,
    search_sku_demands,
    submit_scheduling_job,
    list_scheduling_jobs,
    get_scheduling_job,
    cancel_scheduling_job,
    list_jobs as list_enum_jobs,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _fmt_dt(val: str) -> str:
    if not val:
        return "—"
    try:
        return datetime.fromisoformat(val.replace("Z", "+00:00")).strftime("%m/%d %H:%M")
    except Exception:
        return val


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render():
    # ── 1. SNAPSHOT SELECTION ─────────────────────────────────────────────
    st.markdown(
        "<h3 style='color:#0046AD;font-size:20px;font-weight:700;margin:28px 0 18px 0;"
        "border-left:4px solid #0046AD;padding-left:12px;'>Step 1 — Select Enumeration Snapshot</h3>",
        unsafe_allow_html=True,
    )
    st.caption("The scheduling optimizer consumes a completed enumeration snapshot as its input. Select the snapshot you want to schedule against, or navigate to the Enumeration Dashboard to generate one.")

    try:
        mixes = search_mixes({})
    except Exception:
        mixes = []

    mix_options = {m.get("_id", f"mix-{i}"): m for i, m in enumerate(mixes)}
    mix_labels = {
        mid: f"{m.get('reqPlant','?')} | {m.get('reqBirdSize','?')} | {mid[:8]}…"
        for mid, m in mix_options.items()
    }

    # Pre-select if routed from enumeration dashboard
    preselected = st.session_state.get("sched_selected_mix_id", "")
    default_idx = (list(mix_options.keys()).index(preselected) + 1) if preselected in mix_options else 0

    selected_mix_id = st.selectbox(
        "Enumeration Snapshot",
        options=[""] + list(mix_options.keys()),
        index=default_idx,
        format_func=lambda x: mix_labels.get(x, x) if x else "— no snapshot selected —",
    )
    selected_mix = mix_options.get(selected_mix_id)

    if selected_mix_id:
        plant = selected_mix.get("reqPlant", "—")
        bird = selected_mix.get("reqBirdSize", "—")
        st.markdown(
            f"<div class='simmons-card'>"
            f"<strong>Selected snapshot:</strong> &nbsp;<code>{selected_mix_id}</code>"
            f"&nbsp;&nbsp; Plant: <strong>{plant}</strong> &nbsp; Bird Size: <strong>{bird}</strong>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── 2. SKU DEMAND REVIEW ──────────────────────────────────────────────
    with st.expander("Step 2 — Review SKU Demand (optional)", expanded=False):
        st.caption("Demand records drive how much of each SKU the scheduler needs to fulfill. These are stored in the Scheduling API.")
        try:
            demands = search_sku_demands({})
        except Exception:
            demands = []

        if demands:
            df_d = pd.DataFrame(demands)
            st.dataframe(df_d, use_container_width=True, height=220)
            st.caption(f"{len(demands)} demand records found.")
        else:
            st.info("No SKU demand records found in the Scheduling API. Demand can be loaded via the API or through future import tooling.")

    st.markdown("---")

    # ── 3. SUBMIT SCHEDULING JOB ──────────────────────────────────────────
    st.markdown(
        "<h3 style='color:#0046AD;font-size:20px;font-weight:700;margin:36px 0 18px 0;"
        "border-left:4px solid #0046AD;padding-left:12px;'>Step 3 — Submit Scheduling Job</h3>",
        unsafe_allow_html=True,
    )
    st.caption("The scheduling optimizer runs as a background worker job. Configure the planning horizon and submit. Results are persisted to MongoDB once the job completes.")

    with st.form("scheduling_run_form"):
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            st.markdown("**Run Identity**")
            default_sched_run_id = f"sched-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M')}"
            sched_run_id = st.text_input("Run ID (label)", value=default_sched_run_id)
        with sc2:
            st.markdown("**Planning Horizon**")
            plan_start = st.date_input(
                "Plan Start Date",
                value=datetime.now(timezone.utc).date(),
                help="First day of the scheduling horizon.",
            )
            horizon_days = st.number_input(
                "Horizon (days)", min_value=1, max_value=90, value=12,
                help="Number of production days to schedule. 12 days is the standard short-term horizon.",
            )
        with sc3:
            st.markdown("**Output Settings**")
            save_csv = st.checkbox("Save outputs as CSV", value=False,
                                   help="Write CSV files to the output directory inside the container.")

        sched_submitted = st.form_submit_button("▶ Submit Scheduling Job", type="primary",
                                                 disabled=(not selected_mix_id))

    if not selected_mix_id:
        st.warning("Select an enumeration snapshot above before submitting a scheduling job.")

    if sched_submitted and selected_mix_id:
        payload: dict = {
            "runId": sched_run_id,
            "planStartDate": plan_start.strftime("%Y-%m-%d"),
            "horizonDays": int(horizon_days),
            "saveCsv": save_csv,
        }
        try:
            result = submit_scheduling_job(payload)
            jid = result.get("jobId") or result.get("_id", "")
            st.session_state["sched_last_submitted_job"] = result
            st.success(f"Scheduling job submitted — Run ID: **{sched_run_id}** &nbsp; Job ID: `{jid}`")
        except APIError as exc:
            if exc.status_code == 409:
                st.error("A scheduling job is already running. Wait for it to finish or cancel it before submitting a new run.")
            else:
                st.error(f"API error {exc.status_code}: {exc.detail}")
        except Exception as exc:
            st.error(f"Failed to submit scheduling job: {exc}")

    if "sched_last_submitted_job" in st.session_state:
        last = st.session_state["sched_last_submitted_job"]
        badge = _status_badge(last.get("status", "pending"))
        st.markdown(
            f"<div class='simmons-card' style='margin-top:8px'>"
            f"<strong>Last Submitted:</strong> {last.get('runId', '—')} &nbsp; {badge}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── 4. JOB STATUS ─────────────────────────────────────────────────────
    st.markdown(
        "<h3 style='color:#0046AD;font-size:20px;font-weight:700;margin:36px 0 18px 0;"
        "border-left:4px solid #0046AD;padding-left:12px;'>Scheduling Job Status</h3>",
        unsafe_allow_html=True,
    )
    _sc1, _sc2 = st.columns([5, 1])
    with _sc2:
        st.button("↻ Refresh", key="sched_refresh_jobs")

    try:
        sched_jobs = list_scheduling_jobs()
    except Exception:
        sched_jobs = []

    if sched_jobs:
        sched_sorted = sorted(sched_jobs, key=lambda j: j.get("createdAt", ""), reverse=True)
        rows = []
        for j in sched_sorted[:10]:
            rows.append({
                "Run ID": j.get("runId", "—"),
                "Status": j.get("status", "—"),
                "Plan Start": j.get("planStartDate", "—"),
                "Horizon (days)": j.get("horizonDays", "—"),
                "Submitted": _fmt_dt(j.get("createdAt", "")),
                "Finished": _fmt_dt(j.get("finishedAt", "")),
                "_jobId": j.get("jobId") or j.get("_id", ""),
            })
        df_sj = pd.DataFrame(rows)
        st.dataframe(df_sj.drop(columns=["_jobId"]), use_container_width=True, hide_index=True)

        # Cancel
        running_ids = [r["_jobId"] for r in rows if r["Status"] in ("running", "pending") and r["_jobId"]]
        if running_ids:
            cancel_target = st.selectbox("Cancel a running/pending job", options=["—"] + running_ids, key="sched_cancel_select")
            if st.button("✕ Cancel Selected", type="secondary") and cancel_target != "—":
                try:
                    cancel_scheduling_job(cancel_target)
                    st.success(f"Cancellation requested for `{cancel_target}`.")
                except Exception as exc:
                    st.error(str(exc))
    else:
        st.info("No scheduling jobs found. Submit a run above to get started.")

    st.markdown("---")

    # ── 5. SCHEDULING OUTPUTS / DECISIONS ─────────────────────────────────
    st.markdown(
        "<h3 style='color:#0046AD;font-size:20px;font-weight:700;margin:36px 0 18px 0;"
        "border-left:4px solid #0046AD;padding-left:12px;'>Scheduling Results</h3>",
        unsafe_allow_html=True,
    )
    st.caption("Once a scheduling job completes, its decisions and outputs are stored in the Scheduling API and can be reviewed below.")

    res_tab1, res_tab2 = st.tabs(["Scheduling Decisions", "Scheduling Outputs"])

    with res_tab1:
        try:
            decisions = search_scheduling_decisions({})
        except Exception:
            decisions = []

        if decisions:
            df_dec = pd.DataFrame(decisions)
            st.dataframe(df_dec, use_container_width=True, height=320)
            csv_bytes = df_dec.to_csv(index=False).encode()
            st.download_button("⬇ Export Decisions CSV", data=csv_bytes, file_name="scheduling-decisions.csv", mime="text/csv")
        else:
            st.info("No scheduling decisions found. Submit and complete a scheduling job to populate results.")

    with res_tab2:
        try:
            outputs = search_scheduling_outputs({})
        except Exception:
            outputs = []

        if outputs:
            df_out = pd.DataFrame(outputs)
            st.dataframe(df_out, use_container_width=True, height=320)
            csv_bytes_out = df_out.to_csv(index=False).encode()
            st.download_button("⬇ Export Outputs CSV", data=csv_bytes_out, file_name="scheduling-outputs.csv", mime="text/csv")
        else:
            st.info("No scheduling outputs found.")

    st.caption("Workflow: Select Snapshot → Review Demand → Configure & Submit Job → Monitor Status → Review Decisions & Outputs → Export")

