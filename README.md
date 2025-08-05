# 🧠 ML-Powered Financial Risk Forecasting 📈

This project implements a full-stack financial modeling pipeline using Python — including Monte Carlo simulation, Value at Risk (VaR), LSTM-based stock prediction, and modern portfolio optimization (Efficient Frontier). Built 100% from scratch in GitHub Codespaces.

---

## 📌 Features

- 📊 **Monte Carlo Simulation** for future stock price scenarios
- 📉 **Value at Risk (VaR)** using both Historical and Parametric methods
- 🤖 **LSTM Neural Network** for next-day stock price forecasting
- 💼 **Modern Portfolio Theory** with Efficient Frontier & Sharpe Ratio visualization
- 🛠 Modular, clean Python code across `src/`, with a single entry point: `main.py`
- 🧪 Built for learning, backtesting, and resume projects

---

## 📁 Project Structure

ml-finance-project/
├── data/ <- Optional downloaded datasets
├── results/ <- Simulation charts + portfolio plots
├── notebooks/ <- Optional: exploratory notebooks
├── src/ <- All core logic split into clean modules
│ ├── data_loader.py
│ ├── mc_simulation.py
│ ├── var_calc.py
│ ├── lstm_model.py
│ └── portfolio_opt.py
├── main.py <- 🔁 Main script that runs everything
├── requirements.txt
└── README.md

yaml
Copy
Edit

---

## 🚀 How to Run

### ✅ 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/ml-finance-project.git
cd ml-finance-project
✅ 2. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
✅ 3. Run the full pipeline
bash
Copy
Edit
python main.py
You'll find generated charts in the results/ folder.

📊 Example Outputs
Monte Carlo Simulation

Efficient Frontier

💡 Models & Concepts Used
Model	Description
Monte Carlo	Randomly simulates future stock paths using log returns
VaR	Calculates potential max loss at 95% confidence
LSTM	Predicts future stock prices using past 60-day window
Portfolio Theory	Finds optimal asset weights for Sharpe Ratio

📚 Requirements
Python 3.8+

tensorflow, pandas, yfinance, numpy, matplotlib, scikit-learn

(Install automatically with requirements.txt)

📈 Example Use Cases
Research projects in finance/ML

Investment strategy prototyping

Resume booster for internships or grad school

Training ground for quant interviews

🧠 Author
Wonsang Chang
💼 UC Berkeley | ML & Fintech Focus
📧 [Email/contact here if desired]
🌐 Built during military service using GitHub Codespaces

🏁 Next Steps
 Add PDF report or research write-up

 Convert to Streamlit dashboard

 Add hyperparameter tuning for LSTM