🏈 NFL Moneyline Predictor
This project uses machine learning to predict the winners of upcoming NFL matchups based on team performance statistics and betting spread lines. Built with Python, XGBoost, and Streamlit, it provides an interactive dashboard to explore predictions week by week.

🔍 Overview
The model is trained on historical NFL data from 2020–2024 and uses team-level statistics from 2022–2024 to simulate predictions for the 2025 season. It incorporates:

Weighted team stats (favoring recent seasons)

Spread line data

Yardage and turnover differentials

Confidence scoring for each prediction

Users can interactively select a week and filter predictions by confidence level.

⚙️ Installation
1. Clone the repository
bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPO.git
cd YOUR-REPO
2. Create a virtual environment (optional but recommended)
bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install dependencies
bash
pip install -r requirements.txt
If you don’t have a requirements.txt, use:

bash
pip install streamlit xgboost pandas scikit-learn nfl_data_py
🚀 Running the App
bash
streamlit run model.py
This will launch the dashboard in your browser. You can select a week and adjust the confidence threshold to view predictions.

📊 Features
Predicts winners for upcoming 2025 NFL games

Uses weighted team stats from 2022–2024

Interactive week selector and confidence slider

Displays recommended moneyline pick and model confidence

🧠 Model Details
Algorithm: XGBoost Classifier

Training Data: Historical NFL regular season games (2020–2024)

Features Used:

Fantasy points

Total yardage (passing + rushing + receiving)

Turnovers

Spread line

Yardage differential

Turnover differential

🌐 Deployment
To deploy this app publicly, you can use:

Streamlit Cloud

Render

Heroku

GitHub Pages (for static content only)
