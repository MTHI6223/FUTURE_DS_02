📉 Telco Customer Churn Analysis
> Future Interns — Data Science & Analytics Task 2

A end-to-end customer retention analysis built in Python, using the Kaggle Telco Customer Churn dataset. The project explores why customers leave, which segments are most at risk, and what a business can actually do to reduce churn.

---

📁 Project Structure

```
telco-churn-analysis/
│
├── telco_churn_analysis.py         
├── WA_Fn-UseC_-Telco-Customer-Churn.csv  
├── churn_dashboard.png              
├── churn_deepdive.png              
└── README.md                        
```

---

🎯 Objective

To help a subscription-based telecom business answer four key questions:

- Why are customers leaving?
- Which segments are most likely to churn?
- How long do customers typically stay before leaving?
- What actions can improve retention?

---

📊 Dataset

Source: [Kaggle — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

| Detail | Value |
| Rows | 7,043 customers |
| Columns | 21 features |

---

🔍 Key Findings

| Segment                         | Churn Rate |
| Month-to-month contracts         | 42.7% |
| One-year contracts               | 11.3% |
| Two-year contracts               | 2.8% |
| Fiber optic customers             | 41.9% |
| Electronic check payers           | 45.3% |
| Auto-pay customers               | ~16% |
| Customers aged 0–6 months                 | 53.3% |
| High-risk segment (M2M + Fiber + <12m)   | 70.2% |

The single biggest insight: contract type is the most powerful predictor of churn. A month-to-month customer is 15x more likely to leave than someone on a two-year plan.

---

💡 Business Recommendations

1. Incentivise contract upgrades — offer a discount to lock month-to-month customers into annual plans, especially in their first 12 months
2. Investigate Fiber Optic quality— premium price, double the churn rate of DSL; something is off
3. Bundle add-ons into onboarding — customers with OnlineSecurity and TechSupport churn at ~15% vs ~42% without them
4. Fix the first 6 months — over half of all churned customers leave within 6 months; a 90-day onboarding programme could make a real dent
5. Migrate electronic check users to auto-pay — a small billing credit could shift behaviour and cut churn significantly

---

🛠️ Tech Stack

- Python 3.x
- Pandas — data cleaning and aggregation
- NumPy — numerical operations
- Matplotlib — all charts and the dashboard layout

---

🚀 How to Run
1. Clone the repo
```
clone repository
cd telco-churn-analysis
```

2. Install dependencies
```bash
pip install pandas matplotlib numpy
```

3. Download the dataset

Get the CSV from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place it in the project folder as:
```
WA_Fn-UseC_-Telco-Customer-Churn.csv
```

4. Run the script
```bash
python telco_churn_analysis.py
```

This will generate `churn_dashboard.png`, `churn_deepdive.png`, and print a summary to your terminal.

---

📸 Output Preview

Main Dashboard
[Churn Dashboard](churn_dashboard.png)

Demographic Deep-Dive
[Churn Deep-Dive](churn_deepdive.png)

---

👤 Author

Nkosinathi Mathenjwa
- LinkedIn: [linkedin.com/in/your-profile]([https://linkedin.com/in/your-profile](https://www.linkedin.com/in/nkosinathi-mathenjwa-266ba0235/))
- GitHub: [github.com/your-username]([https://github.com/your-username](https://github.com/MTHI6223))

---

