import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import random
import time
import re
import math

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="KARMIN Autonomous Engine", layout="wide")

# =========================================================
# SESSION STATE
# =========================================================
if 'sovereign_log' not in st.session_state:
    st.session_state.sovereign_log = []
if 'sweep_done' not in st.session_state:
    st.session_state.sweep_done = False
if 'total_saved' not in st.session_state:
    st.session_state.total_saved = 0.0
if 'undo_cache' not in st.session_state:
    st.session_state.undo_cache = []

# =========================================================
# STEP 1: UNIVERSAL INGESTOR  (UNCHANGED)
# =========================================================
class KarminUniversalServiceIngestor:
    def __init__(self, provider_name="AWS", linked_account_id=None):
        self.provider = provider_name
        self.linked_account_id = linked_account_id

    def normalize_service_matrix(self, raw_payload_df):
        timestamp = datetime.now(timezone.utc).isoformat()
        grouped = raw_payload_df.groupby("ServiceName", as_index=False).agg({
            "Cost_USD": "sum",
            "CPU_Percent": "mean",
            "Traffic": "sum",
            "Actual_Val": "mean",
            "Provisioned_Cap": "mean",
            "Dependency": "max",
            "Instance_ID": "first"
        })
        grouped["Timestamp"] = timestamp
        grouped["Provider"] = self.provider
        grouped["LinkedAccountID"] = self.linked_account_id
        return grouped

# =========================================================
# STEP 2: CONTEXT PROFILER  (UNCHANGED)
# =========================================================
class KarminContextProfiler:
    def __init__(self):
        self.threshold_config = {
            "AmazonEC2": {"metric": "CPU_Percent",       "healthy_floor_pct": 30.0, "critical_floor_pct": 10.0},
            "AmazonRDS": {"metric": "Connections",       "healthy_floor_pct": 40.0, "critical_floor_pct": 15.0},
            "AmazonS3":  {"metric": "Access_Frequency",  "healthy_floor_pct": 20.0, "critical_floor_pct":  5.0}
        }

    def get_service_config(self, service):
        return self.threshold_config.get(service, {
            "metric": "Generic_Utilization",
            "healthy_floor_pct": 25.0,
            "critical_floor_pct": 10.0
        })

    def compute_utilization_pct(self, actual_val, provisioned_cap):
        if provisioned_cap <= 0:
            return 0.0
        return max(0.0, min((actual_val / provisioned_cap) * 100, 100.0))

    def compute_inefficiency_pct(self, utilization_pct, healthy_floor_pct):
        if utilization_pct >= healthy_floor_pct:
            return 0.0
        return ((healthy_floor_pct - utilization_pct) / healthy_floor_pct) * 100.0

    def compute_potential_waste_usd(self, cost_usd, inefficiency_pct):
        return cost_usd * (inefficiency_pct / 100.0)

    def compute_uer(self, utilization_pct, cost_usd):
        if cost_usd <= 0:
            return 0.0
        return utilization_pct / cost_usd

    def classify_status(self, utilization_pct, healthy_floor_pct, critical_floor_pct):
        if utilization_pct < critical_floor_pct:
            return "CRITICAL_WASTE_CANDIDATE"
        elif utilization_pct < healthy_floor_pct:
            return "WATCH"
        return "HEALTHY"

    def profile_batch(self, raw_payload_df):
        results = []
        for _, row in raw_payload_df.iterrows():
            config = self.get_service_config(row["ServiceName"])
            util   = self.compute_utilization_pct(row["Actual_Val"], row["Provisioned_Cap"])
            ineff  = self.compute_inefficiency_pct(util, config["healthy_floor_pct"])
            waste  = self.compute_potential_waste_usd(row["Cost_USD"], ineff)
            uer    = self.compute_uer(util, row["Cost_USD"])
            status = self.classify_status(util, config["healthy_floor_pct"], config["critical_floor_pct"])
            results.append({
                "Timestamp":          row["Timestamp"],
                "ServiceName":        row["ServiceName"],
                "Instance_ID":        row["Instance_ID"],
                "Cost_USD":           round(row["Cost_USD"], 2),
                "CPU_Percent":        round(row["CPU_Percent"], 2),
                "Traffic":            round(row["Traffic"], 2),
                "Actual_Val":         row["Actual_Val"],
                "Provisioned_Cap":    row["Provisioned_Cap"],
                "Utilization_Pct":    round(util, 2),
                "Inefficiency_Pct":   round(ineff, 2),
                "Potential_Waste_USD":round(waste, 2),
                "UER":                round(uer, 4),
                "Dependency":         row["Dependency"],
                "Status":             status
            })
        return pd.DataFrame(results)

# =========================================================
# STEP 3: ENSEMBLE ANOMALY ENGINE  (UNCHANGED)
# =========================================================
class KarminEnsembleEngine:
    def __init__(self, z_threshold=2.0, newton_threshold=1.25, tpmad_threshold=3.0):
        self.z_threshold      = z_threshold
        self.newton_threshold = newton_threshold
        self.tpmad_threshold  = tpmad_threshold

    def z_score_detector(self, values):
        baseline = values[:-1]
        mean_val = np.mean(baseline)
        std_val  = np.std(baseline)
        if std_val == 0:
            return 0.0, False
        z = (values[-1] - mean_val) / std_val
        return round(z, 4), z > self.z_threshold

    def newton_interpolation(self, x_values, y_values, x_target):
        n    = len(x_values)
        coef = np.zeros([n, n])
        coef[:, 0] = y_values
        for j in range(1, n):
            for i in range(n - 1, j - 1, -1):
                denom = x_values[i] - x_values[i - j]
                if denom == 0:
                    return float(np.mean(y_values))
                coef[i, j] = (coef[i, j - 1] - coef[i - 1, j - 1]) / denom
        prediction = coef[n - 1, n - 1]
        for i in range(n - 2, -1, -1):
            prediction = coef[i, i] + (x_target - x_values[i]) * prediction
        return float(prediction)

    def newton_detector(self, values):
        baseline  = values[:-1]
        x         = np.arange(len(baseline))
        y         = np.array(baseline, dtype=float)
        predicted = self.newton_interpolation(x, y, len(baseline))
        ratio     = values[-1] / predicted if predicted > 0 else np.inf
        return round(predicted, 4), round(ratio, 4), ratio > self.newton_threshold

    def tpmad_detector(self, values):
        baseline   = np.array(values[:-1], dtype=float)
        median_val = np.median(baseline)
        mad        = np.median(np.abs(baseline - median_val))
        if mad == 0:
            return 0.0, False
        score = abs(values[-1] - median_val) / mad
        return round(score, 4), score > self.tpmad_threshold

    def evaluate_service(self, service_history_df):
        values = service_history_df["Potential_Waste_USD"].tolist()
        if len(values) < 5:
            return None
        latest                          = service_history_df.iloc[-1]
        z_score,    z_flag              = self.z_score_detector(values)
        predicted,  ratio, newton_flag  = self.newton_detector(values)
        tpmad_score, tpmad_flag         = self.tpmad_detector(values)
        votes = sum([z_flag, newton_flag, tpmad_flag])
        return {
            "ServiceName":      latest["ServiceName"],
            "Instance_ID":      latest["Instance_ID"],
            "Cost_USD":         latest["Cost_USD"],
            "CPU_Percent":      latest["CPU_Percent"],
            "Traffic":          latest["Traffic"],
            "Potential_Waste_USD": latest["Potential_Waste_USD"],
            "UER":              latest["UER"],
            "Dependency":       latest["Dependency"],
            "ZScore":           z_score,
            "Newton_Prediction":predicted,
            "Newton_Ratio":     ratio,
            "TPMAD_Score":      tpmad_score,
            "Z_Flag":           z_flag,
            "Newton_Flag":      newton_flag,
            "TPMAD_Flag":       tpmad_flag,
            "Vote_Count":       votes,
            "Anomaly":          votes >= 2
        }

# =========================================================
# STEP 4A: SOVEREIGN PHYSICS ENGINE  (from AI Agent file)
# =========================================================
def get_confidence(sensors: dict) -> float:
    """Weighted sigmoid fusion of Z-score, Newton slope, TPMAD."""
    weights = {'z_score': 1.0, 'slope': 0.8, 'tpmad': 1.2}
    total_w = sum(weights.values())
    score   = 0.0
    for k, v in sensors.items():
        intensity = 1.0 / (1.0 + math.exp(-2.0 * (v - 2.0)))
        score    += intensity * weights.get(k, 0)
    return round(min(1.0, score / total_w), 4)

def get_dependency_risk(inbound: int, outbound: int) -> float:
    """Inbound hub connections = high blast radius."""
    gravity = (inbound * 5.0) + (outbound * 1.0)
    try:
        risk = 1.0 / (1.0 + math.exp(-1.2 * (gravity - 2.0)))
    except OverflowError:
        risk = 1.0
    return round(min(1.0, max(0.0, risk)), 4)

def get_savings_impact(savings: float) -> float:
    """Log-normalized savings saliency. Ignores noise under $10."""
    if savings < 10:
        return 0.0
    return round(min(1.0, math.log10(savings) / 3.0), 4)

def get_rollback_readiness(has_snap: bool) -> float:
    """Binary safety gate: Two-Way Door = 1.0, One-Way Door = 0.0."""
    return 1.0 if has_snap else 0.0

# =========================================================
# STEP 4B: KARMIN SOVEREIGN AGENT  (from AI Agent file — UI-adapted)
# =========================================================
class KarminSovereignAgent:
    """
    Replaces KarminBrain + HITL panel entirely.
    Runs autonomously. Returns structured result dicts for Streamlit UI.
    Authority Vector: A = C * (1 - Rs) * S * Roll
    """
    def __init__(self):
        self.total_saved_this_pass = 0.0

    def _build_sensor_packet(self, ensemble_row: dict) -> dict:
        """
        Bridges Step 3 ensemble outputs → Sovereign Agent sensor format.
        Maps: ZScore → z_score, Newton_Ratio → slope, TPMAD_Score → tpmad
        Maps: Dependency bool → inbound/outbound counts
        Assumes snapshot exists for EC2/RDS (mock); not for S3.
        """
        dep     = ensemble_row.get("Dependency", False)
        service = ensemble_row.get("ServiceName", "")
        return {
            "sensors": {
                "z_score": ensemble_row.get("ZScore", 0.0),
                "slope":   ensemble_row.get("Newton_Ratio", 0.0),
                "tpmad":   ensemble_row.get("TPMAD_Score", 0.0)
            },
            "inbound":      2 if dep else 0,
            "outbound":     1 if dep else 1,
            "savings":      round(ensemble_row.get("Potential_Waste_USD", 0.0) * 24 * 30, 2),
            "has_snapshot": False if "S3" in service else True
        }

    def evaluate_and_execute(self, ensemble_row: dict) -> dict:
        """
        Full sovereign pass for one instance.
        Returns a result dict for UI rendering.
        """
        r_id = ensemble_row.get("Instance_ID", "unknown")
        data = self._build_sensor_packet(ensemble_row)

        # --- Authority Vector ---
        C    = get_confidence(data["sensors"])
        Rs   = get_dependency_risk(data["inbound"], data["outbound"])
        S    = get_savings_impact(data["savings"])
        Roll = get_rollback_readiness(data["has_snapshot"])
        A    = round(C * (1.0 - Rs) * S * Roll, 4)

        # --- Sovereign Decision Logic ---
        if Roll == 0.0:
            status = "BLOCKED"
            reason = (
                f"No snapshot exists for `{r_id}`. "
                f"KARMIN does not execute irreversible actions. "
                f"Create a backup — this instance will auto-terminate on next sweep."
            )
        elif A > 0.5:
            status = "TERMINATED"
            self.total_saved_this_pass += data["savings"]
            reason = (
                f"Confidence **{int(C*100)}%** (Z + Newton + TPMAD consensus). "
                f"Dependency blast radius: **{int(Rs*100)}%** (isolated node). "
                f"Authority Vector A = **{A}** exceeded threshold 0.5. "
                f"Instance terminated. Rollback ready."
            )
        elif C > 0.7 and Rs > 0.4 and Roll == 1.0:
            status = "RESIZED"
            self.total_saved_this_pass += data["savings"]
            reason = (
                f"Confidence **{int(C*100)}%** but dependency risk **{int(Rs*100)}%** too high for termination. "
                f"Instance is a structural hub — downsized safely. "
                f"Authority Vector A = **{A}**."
            )
        else:
            status = "IGNORED"
            reason = (
                f"Authority Vector A = **{A}** below all action thresholds. "
                f"Confidence: {int(C*100)}%, Savings impact: {round(S,2)}, Risk: {int(Rs*100)}%. "
                f"Monitoring continues."
            )

        return {
            "id":           r_id,
            "service":      ensemble_row.get("ServiceName", ""),
            "status":       status,
            "reason":       reason,
            "savings":      data["savings"],
            "C":            C,
            "Rs":           Rs,
            "S":            S,
            "Roll":         Roll,
            "A":            A,
            "anomaly":      ensemble_row.get("Anomaly", False),
            "vote_count":   ensemble_row.get("Vote_Count", 0),
            "z_score":      ensemble_row.get("ZScore", 0.0),
            "newton_ratio": ensemble_row.get("Newton_Ratio", 0.0),
            "tpmad":        ensemble_row.get("TPMAD_Score", 0.0),
            "cpu":          ensemble_row.get("CPU_Percent", 0.0),
            "uer":          ensemble_row.get("UER", 0.0),
        }

    def run_sweep(self, ensemble_df: pd.DataFrame) -> list:
        """Run a full sovereign sweep across all detected anomalies."""
        self.total_saved_this_pass = 0.0
        log = []
        for _, row in ensemble_df.iterrows():
            if row.get("Anomaly", False):
                result = self.evaluate_and_execute(row.to_dict())
                log.append(result)
        return log

# =========================================================
# STEP 5: NARRATOR  (used for NLP tab)
# =========================================================
class KarminNarrator:
    def __init__(self):
        self.openers      = ["Infrastructure Audit:", "Financial Leakage Identified:", "ROI Alert:", "Critical Anomaly:"]
        self.impact_verbs = ["hemorrhaging", "wasting", "draining", "bleeding"]
        self.conclusions  = ["Autonomous optimization justified.", "Capital recovery approved.", "Safe remediation path identified."]

    def generate_explanation(self, result: dict) -> str:
        if result.get("status") == "IGNORED":
            return "Systems nominal. KARMIN detected no anomaly strong enough for autonomous intervention."
        inst    = result.get("id", "Unknown")
        action  = result.get("status", "ACTION")
        savings = result.get("savings", 0)
        C       = result.get("C", 0)
        Rs      = result.get("Rs", 0)
        A       = result.get("A", 0)
        opener  = random.choice(self.openers)
        verb    = random.choice(self.impact_verbs)
        end     = random.choice(self.conclusions)
        return (
            f"**{opener}** Node `{inst}` is {verb} approximately **${savings}/mo**. "
            f"KARMIN computed Confidence={int(C*100)}%, Dependency Risk={int(Rs*100)}%, "
            f"Authority Vector A={A}. Autonomous action **{action}** executed. {end}"
        )

# =========================================================
# STEP 6: ACTUATOR  (UNCHANGED — mock execution layer)
# =========================================================
class KarminActuator:
    def execute_bulk(self, instances):
        time.sleep(0.5)
        return [{"id": i["id"], "action": i["status"]} for i in instances]

    def revert_state(self, undo_cache):
        time.sleep(0.5)
        return "Infrastructure restored."

# =========================================================
# STEP 7: NLP  (UNCHANGED)
# =========================================================
def process_nlp(user_text: str, sovereign_log: list, narrator: KarminNarrator) -> dict:
    text        = user_text.lower()
    instance_id = re.search(r"i-[a-z0-9]{6,20}", text)
    instance_id = instance_id.group(0) if instance_id else None

    if any(w in text for w in ["bill", "cost", "spending", "money", "saved"]):
        total = sum(r["savings"] for r in sovereign_log if r["status"] in ["TERMINATED", "RESIZED"])
        return {"human_summary": f"Autonomous sweep complete. KARMIN recovered **${total:.2f}/mo** across {len(sovereign_log)} evaluated instances."}

    if instance_id:
        item = next((r for r in sovereign_log if r["id"] == instance_id), None)
        if item:
            return {"human_summary": narrator.generate_explanation(item)}

    return {"human_summary": "Incomplete telemetry request. Provide an Instance ID or ask about the bill."}

# =========================================================
# MOCK DATA  (UNCHANGED)
# =========================================================
def generate_mock_raw_payload():
    return pd.DataFrame([
        {"ServiceName": "AmazonEC2", "Cost_USD": round(random.uniform(40, 50), 2), "CPU_Percent": random.randint(2, 15),  "Traffic": random.randint(300, 700),  "Actual_Val": 5,  "Provisioned_Cap": 100, "Dependency": False, "Instance_ID": "i-0abc123"},
        {"ServiceName": "AmazonRDS", "Cost_USD": round(random.uniform(18, 22), 2), "CPU_Percent": random.randint(20, 50), "Traffic": random.randint(200, 500),  "Actual_Val": 25, "Provisioned_Cap": 100, "Dependency": False, "Instance_ID": "i-0def456"},
        {"ServiceName": "AmazonS3",  "Cost_USD": round(random.uniform(11, 13), 2), "CPU_Percent": random.randint(60, 90), "Traffic": random.randint(800, 1500), "Actual_Val": 85, "Provisioned_Cap": 100, "Dependency": True,  "Instance_ID": "i-0ghi789"},
    ])

def generate_mock_history(current_profile_df):
    records = []
    for _, row in current_profile_df.iterrows():
        for _ in range(4):
            baseline_waste = max(0.5, row["Potential_Waste_USD"] * random.uniform(0.15, 0.4))
            records.append({
                "ServiceName":         row["ServiceName"],
                "Instance_ID":         row["Instance_ID"],
                "Cost_USD":            row["Cost_USD"] * random.uniform(0.9, 1.0),
                "CPU_Percent":         row["CPU_Percent"] * random.uniform(1.0, 1.2),
                "Traffic":             row["Traffic"] * random.uniform(0.8, 1.1),
                "Potential_Waste_USD": baseline_waste,
                "UER":                 row["UER"] * random.uniform(1.2, 1.6),
                "Dependency":          row["Dependency"]
            })
        records.append(row.to_dict())
    return pd.DataFrame(records)

# =========================================================
# APP EXECUTION
# =========================================================
st.title("☁️ KARMIN Autonomous Cloud Cost Intelligence")
st.caption("Fully autonomous FinOps agent · No human intervention required")

strategy = st.sidebar.radio("Optimization Strategy", ["COST", "GREEN"])
if st.sidebar.button("🔄 Re-run Sovereign Sweep"):
    st.session_state.sweep_done   = False
    st.session_state.sovereign_log = []
    st.session_state.total_saved   = 0.0
    st.rerun()

# ----------------------------------------------------------
# Steps 1 + 2
# ----------------------------------------------------------
raw_payload   = generate_mock_raw_payload()
ingestor      = KarminUniversalServiceIngestor(provider_name="AWS", linked_account_id="123456789012")
normalized_df = ingestor.normalize_service_matrix(raw_payload)
profiler      = KarminContextProfiler()
profile_df    = profiler.profile_batch(normalized_df)

# ----------------------------------------------------------
# Step 3
# ----------------------------------------------------------
history_df = generate_mock_history(profile_df)
ensemble   = KarminEnsembleEngine()
results    = []
for svc in history_df["ServiceName"].unique():
    out = ensemble.evaluate_service(history_df[history_df["ServiceName"] == svc])
    if out:
        results.append(out)
ensemble_df = pd.DataFrame(results)

# ----------------------------------------------------------
# Step 4: Sovereign Agent — runs once per sweep, cached in session
# ----------------------------------------------------------
narrator  = KarminNarrator()
actuator  = KarminActuator()
agent     = KarminSovereignAgent()

if not st.session_state.sweep_done:
    with st.spinner("⚡ KARMIN Sovereign Sweep in progress..."):
        log = agent.run_sweep(ensemble_df)
        actuator.execute_bulk(log)           # mock execution
        st.session_state.sovereign_log = log
        st.session_state.total_saved   = agent.total_saved_this_pass
        st.session_state.sweep_done    = True

sovereign_log = st.session_state.sovereign_log

# =========================================================
# TOP DASHBOARD
# =========================================================
terminated = [r for r in sovereign_log if r["status"] == "TERMINATED"]
resized    = [r for r in sovereign_log if r["status"] == "RESIZED"]
blocked    = [r for r in sovereign_log if r["status"] == "BLOCKED"]

col1, col2, col3, col4 = st.columns(4)
col1.metric("💰 Monthly Savings Recovered",  f"${st.session_state.total_saved:.2f}")
col2.metric("✅ Instances Terminated",        len(terminated))
col3.metric("🔁 Instances Resized",           len(resized))
col4.metric("🚫 Actions Blocked (No Snap)",   len(blocked))
st.divider()

# =========================================================
# INCOMING CLOUD METRICS  (UNCHANGED)
# =========================================================
st.subheader("📊 Incoming Cloud Metrics")
st.dataframe(profile_df[[
    "ServiceName", "Cost_USD", "CPU_Percent", "Traffic",
    "Utilization_Pct", "Potential_Waste_USD", "UER", "Status"
]], use_container_width=True)

# =========================================================
# SOVEREIGN EXECUTION LOG  (REPLACES HITL PANEL + ANALYSIS PANEL)
# =========================================================
st.subheader("🤖 KARMIN Sovereign Execution Log")
st.caption("All actions below were executed autonomously. No human approval was required.")

for r in sovereign_log:
    status  = r["status"]
    r_id    = r["id"]
    savings = r["savings"]
    reason  = r["reason"]
    A       = r["A"]
    C       = r["C"]
    Rs      = r["Rs"]
    votes   = r["vote_count"]

    if status == "TERMINATED":
        st.success(
            f"✅ **TERMINATED** · `{r_id}` ({r['service']})\n\n"
            f"**Reason:** {reason}\n\n"
            f"**Monthly Savings Recovered:** ${savings:.2f} &nbsp;|&nbsp; "
            f"**Authority Vector A:** {A} &nbsp;|&nbsp; "
            f"**Detector Consensus:** {votes}/3 &nbsp;|&nbsp; "
            f"**Confidence:** {int(C*100)}% &nbsp;|&nbsp; "
            f"**Dependency Risk:** {int(Rs*100)}%"
        )

    elif status == "RESIZED":
        st.info(
            f"🔁 **RESIZED** · `{r_id}` ({r['service']})\n\n"
            f"**Reason:** {reason}\n\n"
            f"**Monthly Savings Recovered:** ${savings:.2f} &nbsp;|&nbsp; "
            f"**Authority Vector A:** {A} &nbsp;|&nbsp; "
            f"**Detector Consensus:** {votes}/3 &nbsp;|&nbsp; "
            f"**Confidence:** {int(C*100)}% &nbsp;|&nbsp; "
            f"**Dependency Risk:** {int(Rs*100)}%"
        )

    elif status == "BLOCKED":
        st.error(
            f"🚫 **BLOCKED** · `{r_id}` ({r['service']})\n\n"
            f"**Reason:** {reason}\n\n"
            f"**Potential Waste (unrecovered):** ${savings:.2f}/mo &nbsp;|&nbsp; "
            f"**Authority Vector A:** {A} &nbsp;|&nbsp; "
            f"**Fix:** Create a snapshot → KARMIN will auto-terminate on next sweep."
        )

    else:
        st.warning(
            f"👁️ **MONITORING** · `{r_id}` ({r['service']})\n\n"
            f"**Reason:** {reason}"
        )

# =========================================================
# EFFICIENCY PANEL  (UNCHANGED)
# =========================================================
st.subheader("💡 Efficiency Analysis")
for _, row in profile_df.iterrows():
    efficiency = row["Traffic"] / row["Cost_USD"] if row["Cost_USD"] > 0 else 0
    st.write(f"**{row['ServiceName']}** → Efficiency Score: `{efficiency:.2f}`")
    if efficiency < 15:
        st.warning(f"{row['ServiceName']}: Low efficiency → Reduce / Resize")
    else:
        st.success(f"{row['ServiceName']}: High efficiency → Protect")

# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3 = st.tabs(["🛡️ Triage", "🧠 Ensemble Engine", "💻 NLP Terminal"])

with tab1:
    st.caption("Detailed audit cards for each sovereign decision.")
    for r in sovereign_log:
        label = f"{r['status']} · {r['id']} · ${r['savings']:.2f}/mo"
        with st.expander(label):
            st.markdown(f"**Service:** {r['service']}")
            st.markdown(f"**Status:** `{r['status']}`")
            st.markdown(f"**Reason:** {r['reason']}")
            st.json({
                "Authority_Vector_A":  r["A"],
                "Confidence_C":        r["C"],
                "Dependency_Risk_Rs":  r["Rs"],
                "Savings_Impact_S":    r["S"],
                "Rollback_Readiness":  r["Roll"],
                "Detector_Votes":      f"{r['vote_count']}/3",
                "ZScore":              r["z_score"],
                "Newton_Ratio":        r["newton_ratio"],
                "TPMAD":               r["tpmad"],
                "CPU_Percent":         r["cpu"],
                "UER":                 r["uer"],
            })

with tab2:
    st.dataframe(ensemble_df, use_container_width=True)

with tab3:
    query = st.chat_input("Command Karmin (e.g., 'Check i-0abc123' or 'how much did we save?')")
    if query:
        st.chat_message("user").write(query)
        res = process_nlp(query, sovereign_log, narrator)
        st.chat_message("assistant").write(res["human_summary"])