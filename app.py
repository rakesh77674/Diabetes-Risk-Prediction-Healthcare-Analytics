"""
Diabetes Risk Prediction - Real-Time Dashboard
Run:  streamlit run app.py
Needs: diabetes_binary_health.csv next to this file (or set the path in the sidebar).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
TARGET = "Diabetes_binary"

AGE_LABELS = {1: "18-24", 2: "25-29", 3: "30-34", 4: "35-39", 5: "40-44",
              6: "45-49", 7: "50-54", 8: "55-59", 9: "60-64", 10: "65-69",
              11: "70-74", 12: "75-79", 13: "80+"}
GENHLTH = {1: "Excellent", 2: "Very good", 3: "Good", 4: "Fair", 5: "Poor"}
EDU = {1: "None / kindergarten", 2: "Elementary", 3: "Some high school",
       4: "High school grad", 5: "Some college", 6: "College grad"}
INCOME = {1: "< $10k", 2: "$10-15k", 3: "$15-20k", 4: "$20-25k",
          5: "$25-35k", 6: "$35-50k", 7: "$50-75k", 8: "$75k+"}

st.set_page_config(page_title="Diabetes Risk Dashboard", page_icon="🩺", layout="wide")


# ----------------------------------------------------------------- data/models
@st.cache_data(show_spinner="Loading data...")
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path).drop_duplicates().reset_index(drop=True)


@st.cache_resource(show_spinner="Training models (first run only)...")
def train_all(path: str):
    df = load_data(path)
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(class_weight="balanced", max_iter=1000,
                                         random_state=RANDOM_STATE))]),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced", max_depth=8, min_samples_leaf=20,
            random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, max_depth=12, min_samples_leaf=5,
            class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE),
    }
    rows, probs = [], {}
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        p = m.predict_proba(X_te)[:, 1]
        pred = (p >= 0.5).astype(int)
        probs[name] = p
        rows.append({"Model": name,
                     "Accuracy": accuracy_score(y_te, pred),
                     "Precision": precision_score(y_te, pred),
                     "Recall": recall_score(y_te, pred),
                     "F1 Score": f1_score(y_te, pred),
                     "ROC-AUC": roc_auc_score(y_te, p),
                     "PR-AUC": average_precision_score(y_te, p)})
    return models, pd.DataFrame(rows), X_te, y_te, probs, list(X.columns)


# --------------------------------------------------------------------- sidebar
st.sidebar.title("🩺 Settings")
from pathlib import Path
default_csv = str(Path(__file__).parent / "diabetes_binary_health.csv")
csv_path = st.sidebar.text_input("Dataset path", default_csv)
if not Path(csv_path).exists():
    st.error(f"Can't find `{csv_path}`. Put the CSV next to app.py or fix the path in the sidebar.")
    st.stop()

df = load_data(csv_path)
models, results_df, X_test, y_test, probs, FEATURES = train_all(csv_path)

model_name = st.sidebar.selectbox("Model", list(models), index=2)
threshold = st.sidebar.slider("High-risk threshold", 0.05, 0.95, 0.50, 0.05,
                              help="Probability above which a person is flagged high-risk.")
model = models[model_name]

st.title("Diabetes Risk Prediction & Healthcare Analytics")
st.caption("Screening-style risk score from survey indicators. Not a medical diagnosis.")

tab_pred, tab_pop, tab_perf, tab_live = st.tabs(
    ["🎯 Live Risk Predictor", "📊 Population Analytics",
     "🤖 Model Performance", "📡 Live Stream (simulated)"])

# ------------------------------------------------------------- 1. live predictor
with tab_pred:
    left, right = st.columns([3, 2])
    with left:
        c1, c2, c3 = st.columns(3)
        bmi = c1.slider("BMI", 12, 60, 27)
        age = c2.select_slider("Age group", options=list(AGE_LABELS), value=7,
                               format_func=AGE_LABELS.get)
        sex = c3.radio("Sex", [0, 1], format_func=lambda v: "Female" if v == 0 else "Male",
                       horizontal=True)

        c1, c2, c3 = st.columns(3)
        genhlth = c1.select_slider("General health", options=list(GENHLTH), value=3,
                                   format_func=GENHLTH.get)
        menthlth = c2.slider("Bad mental-health days (30d)", 0, 30, 0)
        physhlth = c3.slider("Bad physical-health days (30d)", 0, 30, 0)

        c1, c2 = st.columns(2)
        edu = c1.select_slider("Education", options=list(EDU), value=5, format_func=EDU.get)
        income = c2.select_slider("Income", options=list(INCOME), value=6, format_func=INCOME.get)

        st.markdown("**Health & lifestyle**")
        flags = {
            "HighBP": "High blood pressure", "HighChol": "High cholesterol",
            "CholCheck": "Cholesterol check (5y)", "Smoker": "Smoker",
            "Stroke": "Stroke history", "HeartDiseaseorAttack": "Heart disease / attack",
            "PhysActivity": "Physically active", "Fruits": "Eats fruit daily",
            "Veggies": "Eats veg daily", "HvyAlcoholConsump": "Heavy alcohol use",
            "AnyHealthcare": "Has healthcare cover", "NoDocbcCost": "Skipped doctor (cost)",
            "DiffWalk": "Difficulty walking"}
        defaults = {"CholCheck": True, "PhysActivity": True, "Fruits": True,
                    "Veggies": True, "AnyHealthcare": True}
        cols = st.columns(3)
        vals = {}
        for i, (k, label) in enumerate(flags.items()):
            vals[k] = int(cols[i % 3].checkbox(label, value=defaults.get(k, False), key=f"f_{k}"))

    person = {**vals, "BMI": bmi, "Age": age, "Sex": sex, "GenHlth": genhlth,
              "MentHlth": menthlth, "PhysHlth": physhlth, "Education": edu, "Income": income}
    x_row = pd.DataFrame([person])[FEATURES]
    risk = float(model.predict_proba(x_row)[0, 1])

    with right:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=risk * 100, number={"suffix": "%"},
            title={"text": f"Predicted risk ({model_name})"},
            gauge={"axis": {"range": [0, 100]},
                   "bar": {"color": "#d62728" if risk >= threshold else "#2ca02c"},
                   "threshold": {"line": {"color": "black", "width": 3},
                                 "value": threshold * 100}}))
        gauge.update_layout(height=320, margin=dict(t=60, b=0, l=20, r=20))
        st.plotly_chart(gauge, use_container_width=True)
        if risk >= threshold:
            st.error("⚠️ Flagged **high risk** at the current threshold.")
        else:
            st.success("✅ Below the high-risk threshold.")
        st.caption("The models use balanced class weights, so scores run higher than "
                   "true population probabilities. Treat this as a relative risk score.")

        if model_name == "Random Forest":
            imp = pd.Series(model.feature_importances_, index=FEATURES).nlargest(8)
            fig = px.bar(imp[::-1], orientation="h", title="Top drivers (model-wide)")
            fig.update_layout(showlegend=False, height=300, margin=dict(t=40, b=0),
                              xaxis_title=None, yaxis_title=None)
            st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------- 2. population analytics
with tab_pop:
    f1c, f2c = st.columns(2)
    ages = f1c.multiselect("Age groups", list(AGE_LABELS), default=list(AGE_LABELS),
                           format_func=AGE_LABELS.get)
    sexes = f2c.multiselect("Sex", [0, 1], default=[0, 1],
                            format_func=lambda v: "Female" if v == 0 else "Male")
    sub = df[df["Age"].isin(ages) & df["Sex"].isin(sexes)]
    if sub.empty:
        st.warning("No rows match the filters.")
    else:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Records", f"{len(sub):,}")
        k2.metric("Diabetes / pre-diabetes rate", f"{sub[TARGET].mean() * 100:.1f}%")
        k3.metric("Avg BMI", f"{sub['BMI'].mean():.1f}")
        k4.metric("High BP prevalence", f"{sub['HighBP'].mean() * 100:.1f}%")

        a, b = st.columns(2)
        gh = sub.groupby("GenHlth")[TARGET].mean().mul(100).rename(index=GENHLTH).reset_index()
        a.plotly_chart(px.bar(gh, x="GenHlth", y=TARGET, title="Outcome rate by general health",
                              labels={TARGET: "Rate (%)", "GenHlth": ""}),
                       use_container_width=True)
        b.plotly_chart(px.histogram(sub, x="BMI", color=sub[TARGET].map(
            {0: "No diabetes", 1: "Pre-diabetes / diabetes"}), nbins=50, barmode="overlay",
            histnorm="probability density", opacity=0.6, title="BMI distribution by outcome",
            labels={"color": "Outcome"}), use_container_width=True)

        ind = ["HighBP", "HighChol", "HeartDiseaseorAttack", "Stroke",
               "PhysActivity", "Smoker", "DiffWalk"]
        prev = sub.groupby(TARGET)[ind].mean().T.mul(100)
        prev.columns = ["No diabetes", "Pre-diabetes / diabetes"][:len(prev.columns)]
        c, d = st.columns(2)
        c.plotly_chart(px.bar(prev, barmode="group", title="Indicator prevalence (%)"),
                       use_container_width=True)
        age_rate = sub.groupby("Age")[TARGET].mean().mul(100).rename(index=AGE_LABELS).reset_index()
        d.plotly_chart(px.line(age_rate, x="Age", y=TARGET, markers=True,
                               title="Outcome rate by age group",
                               labels={TARGET: "Rate (%)", "Age": ""}),
                       use_container_width=True)

# ------------------------------------------------------- 3. model performance
with tab_perf:
    st.dataframe(results_df.set_index("Model").style.format("{:.3f}"), use_container_width=True)
    a, b = st.columns(2)
    roc = go.Figure()
    for n, p in probs.items():
        fpr, tpr, _ = roc_curve(y_test, p)
        roc.add_scatter(x=fpr, y=tpr, mode="lines", name=f"{n} (AUC {roc_auc_score(y_test, p):.3f})")
    roc.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash"))
    roc.update_layout(title="ROC curves", xaxis_title="False positive rate",
                      yaxis_title="True positive rate")
    a.plotly_chart(roc, use_container_width=True)

    cm = confusion_matrix(y_test, (probs[model_name] >= threshold).astype(int))
    b.plotly_chart(px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                             x=["No diabetes", "Diabetes / pre"], y=["No diabetes", "Diabetes / pre"],
                             labels=dict(x="Predicted", y="Actual"),
                             title=f"Confusion matrix - {model_name} @ {threshold:.2f}"),
                   use_container_width=True)

# ------------------------------------------------------- 4. simulated live feed
with tab_live:
    st.write("Simulates incoming patient records: every few seconds a random record from the "
             "held-out test set is scored by the selected model. In production, replace "
             "`next_record()` with a Kafka / database / API call.")
    c1, c2 = st.columns([1, 3])
    running = c1.toggle("Streaming", value=True)
    if c1.button("Reset feed"):
        st.session_state.pop("feed", None)

    def next_record():
        i = np.random.randint(len(X_test))
        row = X_test.iloc[[i]]
        return row, int(y_test.iloc[i])

    @st.fragment(run_every="2s")
    def live_feed():
        feed = st.session_state.setdefault("feed", [])
        if running:
            row, actual = next_record()
            p = float(model.predict_proba(row)[0, 1])
            feed.append({"time": pd.Timestamp.now().strftime("%H:%M:%S"), "risk": p,
                         "flagged": p >= threshold, "actual": actual,
                         "BMI": float(row["BMI"].iloc[0]),
                         "Age": AGE_LABELS[int(row["Age"].iloc[0])],
                         "GenHlth": GENHLTH[int(row["GenHlth"].iloc[0])]})
            del feed[:-300]  # keep last 300
        if not feed:
            st.info("Waiting for records...")
            return
        live = pd.DataFrame(feed)
        m1, m2, m3 = st.columns(3)
        m1.metric("Records scored", len(live))
        m2.metric("High-risk flagged", f"{live['flagged'].mean() * 100:.0f}%")
        m3.metric("Avg risk (last 20)", f"{live['risk'].tail(20).mean() * 100:.0f}%")
        fig = px.line(live, x="time", y="risk", title="Risk score per incoming record")
        fig.add_hline(y=threshold, line_dash="dash", line_color="red")
        fig.update_layout(yaxis_range=[0, 1], xaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(live.tail(10).iloc[::-1], use_container_width=True, hide_index=True)

    live_feed()