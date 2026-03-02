# AI-Powered Predictive Maintenance System (Executive Demo)

An enterprise-grade predictive maintenance dashboard leveraging **XGBoost** and **SHAP** for early failure detection and explainable AI insights.

## ✨ Features
- **Recall-Optimized XGBoost**: Tuned for 71.8% recall to ensure maximum failure detection.
- **Explainable AI**: Integration with SHAP for real-time risk factor transparency.
- **Executive UI**: 3-tier risk segmentation, system health scores, and 7-day risk trajectory charts.
- **Fleet Analytics**: Aggregate management view of total asset health and risk distribution.
- **Interactive Simulation**: Built-in production simulator for daily fleet scans and automated alerting.

## 🛠️ Tech Stack
- **Engine**: XGBoost, Scikit-learn
- **Explainability**: SHAP (Shapley Additive Explanations)
- **Frontend**: Streamlit
- **Visualization**: Plotly

## 🚀 Getting Started

1. **Clone the repository**:
   ```bash
   git clone https://github.com/lalitshirsath008/predictive-analysis.git
   cd predictive-analysis
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the dashboard**:
   ```bash
   streamlit run app.py
   ```

## 📊 Model Logic
- **Optimized Threshold**: 0.23 (Optimized for Recall)
- **Features**: Air Temperature, Process Temperature, Rotational Speed, Torque, Tool Wear, Machine Type.

---
*Prototype – Internal AI Initiative | RGK Group v3.0*
