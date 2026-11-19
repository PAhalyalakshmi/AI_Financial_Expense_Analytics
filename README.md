
# AI Financial Expense Analytics & Budget Planning Dashboard

## Project Overview

This is a Python-based financial analytics application
designed to analyze income, expenses, savings, and budgets.

The application provides monthly financial summaries,
expense category analysis, budget monitoring, and a
basic Machine Learning-based expense estimation.

## Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Random Forest Regression
- Excel Reporting
- CSV Data Analysis

## Features

1. Upload financial transaction data from CSV or Excel.
2. Automatically categorize transactions using keywords.
3. Calculate total income, expenses, and savings.
4. Calculate savings rate.
5. Analyze monthly income and expense trends.
6. Perform category-wise expense analysis.
7. Compare monthly budgets against actual expenses.
8. Identify over-budget categories.
9. Estimate next-month expenses using Random Forest.
10. Export financial reports to Excel and CSV.

## Installation

Install Python 3.10 or later.

Install dependencies:

    pip install -r requirements.txt

## Run the Application

    streamlit run app.py

## How to Use

1. Upload sample_expenses.csv.
2. Click Analyze Financial Data.
3. Review financial KPIs.
4. Explore monthly expense trends.
5. Set budgets for expense categories.
6. Review budget versus actual spending.
7. Review the next-month expense estimate.
8. Download the Excel or CSV report.

## Business Use Case

This dashboard helps users and finance teams understand
spending patterns, monitor budgets, and prepare financial reports.

## Limitations

- Sample data is synthetic.
- Expense categories use keyword-based rules.
- Budget amounts are entered manually.
- Forecasting is a demonstration, not a validated financial model.
- The application does not provide financial advice.
