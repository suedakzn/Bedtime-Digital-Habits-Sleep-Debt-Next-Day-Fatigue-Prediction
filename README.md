# Bedtime-Digital-Habits-Sleep-Debt-Next-Day-Fatigue-Prediction

An end-to-end behavioral analytics and machine learning project examining how pre-bedtime smartphone usage, blue light exposure, and evening caffeine intake affect sleep architecture and next-day fatigue[cite: 1]. The study integrates **unsupervised behavioral segmentation (K-Means)** for targeted CRM strategies and **supervised predictive modeling (LightGBM)** to forecast fatigue scores[cite: 1].

---

## Project Overview & Objectives
Late-night screen time and evening stimulation disrupt circadian rhythms, increasing sleep latency and compounding sleep debt[cite: 1]. 

This project aims to answer:
1. **Behavioral Patterns:** How do bedtime digital habits, caffeine intake, and chronotypes correlate with sleep quality[cite: 1]?
2. **Behavioral Segmentation:** Can participants be segmented into meaningful personas using unsupervised clustering[cite: 1]?
3. **CRM Strategy:** What data-driven, personalized interventions can be designed for each behavioral segment[cite: 1]?
4. **Fatigue Prediction:** Can next-day fatigue scores (1–10) be predicted accurately using behavioral and physiological metrics[cite: 1]?

---

## Dataset Overview
The dataset contains 8,500 participant observations across 18 behavioral, demographic, and physiological features[cite: 1]:
* **Demographics & Chronotype:** `age`, `gender`, `occupation_type`, `chronotype`[cite: 1]
* **Bedtime & Digital Habits:** `bedtime_phone_minutes`, `screen_brightness_pct`, `blue_light_filter_active`, `primary_bedtime_app`[cite: 1]
* **Lifestyle:** `caffeine_post_5pm_mg`, `physical_activity_min`[cite: 1]
* **Sleep Architecture:** `sleep_latency_min`, `total_sleep_hours`, `deep_sleep_pct`, `rem_sleep_pct`, `morning_alarm_snoozes`[cite: 1]
* **Target & Outcomes:** `next_day_fatigue_score` (1.0–10.0), `sleep_debt_category`[cite: 1]

---

## Key Findings & Methodology

### 1. Behavioral Clustering (K-Means, k=4)
A feature set of 19 standardized numerical and one-hot encoded categorical variables was utilized for segmentation. K-Means separated the cohort into 4 distinct behavioral personas:

| Segment Name | Share | Bedtime Phone (min) | Sleep Duration (hrs) | Sleep Latency (min) | Mean Fatigue (1-10) | Profile & CRM Strategy |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Morning Larks** | 25.5% | ~49.6 | 7.66 | 35.9 | **1.86** | 100% Morning Lark; low latency and high restorative sleep. **CRM:** Routine preservation, positive habit streaks, and morning wellness content. |
| **Balanced Intermediates** | 28.3% | ~45.2 | 6.84 | 33.8 | **2.43** | Consistent sleep duration and moderate phone usage. **CRM:** Consistency challenges and pre-bed wind-down breathing exercises. |
| **Night Owls** | 28.9% | ~48.4 | 5.23 | 35.3 | **4.77** | Fixed morning schedules conflict with late chronotypes, accumulating sleep debt. **CRM:** 17:00 caffeine cut-off notifications and blue-light filter prompts. |
| **High-Risk Screen Users** | 17.3% | **115.9** | **4.79** | **68.5** | **7.67** | Severe sleep latency (over 1 hour), low sleep duration, heavy TikTok/Reels/YouTube use. **CRM:** Digital sunset protocol, auto-dimming brightness after 22:00, and smart escalating alarms. |

---

### 2. Next-Day Fatigue Prediction (Supervised Learning)
Four regression models were trained on 80% of the dataset to predict `next_day_fatigue_score`:

| Model | RMSE | MAE | $R^2$ Score |
| :--- | :---: | :---: | :---: |
| **LightGBM**  | **0.5769** | **0.4139** | **0.9553** |
| **Random Forest** | 0.6036 | 0.4275 | 0.9511 |
| **XGBoost** | 0.6259 | 0.4560 | 0.9474 |
| **Ridge Regression** | 0.7718 | 0.6136 | 0.9201 |

* **Model Insights:** Tree-based gradient boosting models outperformed linear regression by effectively capturing non-linear threshold effects (e.g., screen time worsening fatigue non-linearly once it exceeds 60+ minutes).
* **Feature Importance:** Physiological sleep architecture metrics (**total sleep hours**, **sleep latency**, and **deep sleep %**) were the primary determinants of next-day fatigue[cite: 3]. Upstream digital behaviors (**screen brightness**, **bedtime phone duration**, and **evening caffeine**) were the primary behavioral triggers[cite: 3].
* **Diagnostics & Segment Validation:** Residuals followed an unbiased, zero-centered normal distribution[cite: 4]. Segment-level Mean Absolute Error (MAE) remained consistently low across all groups (0.30 for Morning Larks to 0.50 for Night Owls and High-Risk users).

---

## Tech Stack
* **Language:** Python
* **Data Processing & EDA:** `pandas`, `numpy`
* **Visualization:** `matplotlib`, `seaborn`
* **Machine Learning & Preprocessing:** `scikit-learn` (`StandardScaler`, `KMeans`, `silhouette_score`, `Ridge`, `RandomForestRegressor`)
* **Gradient Boosting:** `lightgbm`, `xgboost`

---

## Repository Structure
```text
├── data/
│   └── bedtime_screentime_sleep_debt.csv       # Dataset
├── notebooks/
│   └── sleep-behavior-crm-fatigue.ipynb        # Main analysis & modeling notebook
├── images/
│   ├── top_feature_importance.png              # Feature importance visualization
│   └── model_diagnostics.png                   # Actual vs predicted & residuals
├── requirements.txt                            # Python environment dependencies
└── README.md                                   # Project documentation
```
---
## Practical Applications & CRM Takeaways
- Personalized Interventions: Generic sleep reminders are ineffective when user chronotypes and habits diverge fundamentally. In-app communication should be segmented by persona.

- Proactive Warning System: Deploying the trained LightGBM model can evaluate evening user telemetry in real time to trigger preventive wind-down nudges before fatigue compounds.
