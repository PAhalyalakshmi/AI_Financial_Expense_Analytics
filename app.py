
import streamlit as st
import pandas as pd
import numpy as np
import io
import re

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI Financial Expense Analytics",
    page_icon="💰",
    layout="wide"
)


# --------------------------------------------------
# EXPENSE CATEGORY KEYWORDS
# --------------------------------------------------

CATEGORY_KEYWORDS = {
    "Food": [
        "restaurant", "food", "grocery", "groceries",
        "cafe", "coffee", "lunch", "dinner",
        "swiggy", "zomato"
    ],

    "Transport": [
        "bus", "train", "metro", "taxi", "uber",
        "ola", "fuel", "petrol", "travel"
    ],

    "Shopping": [
        "shopping", "clothes", "fashion", "amazon",
        "flipkart", "shoes", "dress"
    ],

    "Bills": [
        "electricity", "water", "internet",
        "mobile bill", "recharge", "bill"
    ],

    "Education": [
        "course", "education", "book", "books",
        "college", "tuition", "training"
    ],

    "Healthcare": [
        "hospital", "medicine", "medical",
        "doctor", "pharmacy", "health"
    ],

    "Entertainment": [
        "movie", "cinema", "game", "music",
        "netflix", "entertainment"
    ],

    "Rent": [
        "rent", "house rent", "room rent"
    ],

    "Salary": [
        "salary", "payroll", "monthly salary"
    ],

    "Freelance": [
        "freelance", "project payment",
        "consulting", "client payment"
    ]
}


# --------------------------------------------------
# CLEAN TEXT
# --------------------------------------------------

def clean_text(text):

    if pd.isna(text):
        return ""

    text = str(text).lower()

    text = re.sub(
        r"[^a-zA-Z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# --------------------------------------------------
# CATEGORY CLASSIFICATION
# --------------------------------------------------

def classify_category(description, transaction_type):

    description = clean_text(description)

    transaction_type = str(
        transaction_type
    ).strip().lower()

    if transaction_type == "income":

        for category in ["Salary", "Freelance"]:

            for keyword in CATEGORY_KEYWORDS[category]:

                if keyword in description:
                    return category

        return "Other Income"

    category_scores = {}

    for category, keywords in CATEGORY_KEYWORDS.items():

        if category in ["Salary", "Freelance"]:
            continue

        score = 0

        for keyword in keywords:

            if keyword in description:
                score += 1

        category_scores[category] = score

    if not category_scores:

        return "Other"

    highest_score = max(
        category_scores.values()
    )

    if highest_score == 0:

        return "Other"

    return max(
        category_scores,
        key=category_scores.get
    )


# --------------------------------------------------
# LOAD AND CLEAN DATA
# --------------------------------------------------

def prepare_data(dataframe):

    dataframe = dataframe.copy()

    dataframe.columns = (
        dataframe.columns
        .astype(str)
        .str.strip()
    )

    required_columns = [
        "Date",
        "Description",
        "Amount",
        "Type"
    ]

    missing_columns = [
        column for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing columns: "
            + ", ".join(missing_columns)
        )

    dataframe["Date"] = pd.to_datetime(
        dataframe["Date"],
        errors="coerce"
    )

    dataframe["Amount"] = pd.to_numeric(
        dataframe["Amount"],
        errors="coerce"
    )

    dataframe["Description"] = (
        dataframe["Description"]
        .fillna("")
        .astype(str)
    )

    dataframe["Type"] = (
        dataframe["Type"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.title()
    )

    dataframe = dataframe.dropna(
        subset=["Date", "Amount"]
    )

    dataframe = dataframe[
        dataframe["Amount"] >= 0
    ]

    dataframe = dataframe[
        dataframe["Type"].isin(
            ["Income", "Expense"]
        )
    ]

    dataframe["Category"] = (
        dataframe.apply(
            lambda row: classify_category(
                row["Description"],
                row["Type"]
            ),
            axis=1
        )
    )

    dataframe["Month"] = (
        dataframe["Date"]
        .dt.to_period("M")
        .astype(str)
    )

    return dataframe


# --------------------------------------------------
# MONTHLY ANALYSIS
# --------------------------------------------------

def calculate_monthly_summary(dataframe):

    monthly_income = (
        dataframe[
            dataframe["Type"] == "Income"
        ]
        .groupby("Month")["Amount"]
        .sum()
    )

    monthly_expenses = (
        dataframe[
            dataframe["Type"] == "Expense"
        ]
        .groupby("Month")["Amount"]
        .sum()
    )

    monthly_summary = pd.DataFrame({
        "Income": monthly_income,
        "Expenses": monthly_expenses
    }).fillna(0)

    monthly_summary["Savings"] = (
        monthly_summary["Income"]
        - monthly_summary["Expenses"]
    )

    monthly_summary = (
        monthly_summary
        .sort_index()
    )

    return monthly_summary


# --------------------------------------------------
# MONTHLY EXPENSE FORECAST
# --------------------------------------------------

def forecast_next_month(monthly_summary):

    monthly_expenses = (
        monthly_summary["Expenses"]
        .sort_index()
    )

    if len(monthly_expenses) < 6:

        return None, None, (
            "At least 6 months of expense data "
            "are recommended for this demonstration."
        )

    values = monthly_expenses.values.astype(float)

    feature_rows = []
    target_values = []

    # Use previous 3 months to estimate the following month
    for index in range(3, len(values)):

        feature_rows.append([
            values[index - 1],
            values[index - 2],
            values[index - 3],
            np.mean(values[index - 3:index])
        ])

        target_values.append(
            values[index]
        )

    X = np.array(
        feature_rows
    )

    y = np.array(
        target_values
    )

    if len(X) < 3:

        return None, None, (
            "Not enough historical observations "
            "to train the model."
        )

    # Keep the latest observations for testing
    split_index = max(
        1,
        int(len(X) * 0.8)
    )

    if split_index >= len(X):

        split_index = len(X) - 1

    X_train = X[:split_index]

    X_test = X[split_index:]

    y_train = y[:split_index]

    y_test = y[split_index:]

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    test_predictions = model.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        test_predictions
    )

    # Train on all available historical observations
    model.fit(
        X,
        y
    )

    last_three_months = values[-3:]

    next_features = np.array([[
        last_three_months[-1],
        last_three_months[-2],
        last_three_months[-3],
        np.mean(last_three_months)
    ]])

    forecast = model.predict(
        next_features
    )[0]

    forecast = max(
        0,
        forecast
    )

    return round(forecast, 2), round(mae, 2), None


# --------------------------------------------------
# EXCEL REPORT
# --------------------------------------------------

def create_excel_report(dataframe, monthly_summary):

    output = io.BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl"
    ) as writer:

        dataframe.to_excel(
            writer,
            index=False,
            sheet_name="Transactions"
        )

        monthly_summary.to_excel(
            writer,
            sheet_name="Monthly Summary"
        )

    output.seek(0)

    return output


# --------------------------------------------------
# MAIN APPLICATION
# --------------------------------------------------

st.title(
    "💰 AI Financial Expense Analytics & Budget Planning"
)

st.write(
    "Analyze income, expenses, savings, budgets, "
    "and spending patterns through an interactive dashboard."
)

st.info(
    "This project is an educational financial analytics tool. "
    "It does not provide financial advice."
)

st.divider()


# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------

st.subheader(
    "1. Upload Financial Transactions"
)

uploaded_file = st.file_uploader(
    "Upload CSV or Excel file",
    type=["csv", "xlsx"]
)


if uploaded_file is not None:

    try:

        if uploaded_file.name.endswith(".csv"):

            raw_data = pd.read_csv(
                uploaded_file
            )

        else:

            raw_data = pd.read_excel(
                uploaded_file
            )

        st.success(
            "File uploaded successfully!"
        )

        st.subheader(
            "Transaction Preview"
        )

        st.dataframe(
            raw_data.head(10),
            use_container_width=True
        )

        st.write(
            "Total records:",
            len(raw_data)
        )

        st.divider()


        # --------------------------------------------------
        # ANALYZE DATA
        # --------------------------------------------------

        if st.button(
            "Analyze Financial Data",
            type="primary"
        ):

            try:

                cleaned_data = prepare_data(
                    raw_data
                )

                if cleaned_data.empty:

                    st.error(
                        "No valid financial transactions found."
                    )

                    st.stop()

                monthly_summary = (
                    calculate_monthly_summary(
                        cleaned_data
                    )
                )

                st.session_state[
                    "financial_data"
                ] = cleaned_data

                st.session_state[
                    "monthly_summary"
                ] = monthly_summary

                st.success(
                    "Financial analysis completed!"
                )

            except Exception as error:

                st.error(
                    f"Analysis error: {error}"
                )


    except Exception as error:

        st.error(
            f"Unable to read file: {error}"
        )


# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

if "financial_data" in st.session_state:

    financial_data = st.session_state[
        "financial_data"
    ]

    monthly_summary = st.session_state[
        "monthly_summary"
    ]

    st.divider()

    st.subheader(
        "2. Financial Analytics Dashboard"
    )


    # --------------------------------------------------
    # KPI METRICS
    # --------------------------------------------------

    total_income = financial_data[
        financial_data["Type"] == "Income"
    ]["Amount"].sum()

    total_expenses = financial_data[
        financial_data["Type"] == "Expense"
    ]["Amount"].sum()

    total_savings = (
        total_income - total_expenses
    )

    savings_rate = (
        total_savings / total_income * 100
        if total_income > 0
        else 0
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Income",
        f"₹{total_income:,.2f}"
    )

    col2.metric(
        "Total Expenses",
        f"₹{total_expenses:,.2f}"
    )

    col3.metric(
        "Total Savings",
        f"₹{total_savings:,.2f}"
    )

    col4.metric(
        "Savings Rate",
        f"{savings_rate:.2f}%"
    )


    # --------------------------------------------------
    # MONTHLY TRENDS
    # --------------------------------------------------

    st.divider()

    st.subheader(
        "3. Monthly Income, Expenses & Savings"
    )

    st.line_chart(
        monthly_summary[
            ["Income", "Expenses", "Savings"]
        ]
    )


    # --------------------------------------------------
    # CATEGORY ANALYSIS
    # --------------------------------------------------

    st.divider()

    st.subheader(
        "4. Expense Category Analysis"
    )

    expense_data = financial_data[
        financial_data["Type"] == "Expense"
    ]

    category_expenses = (
        expense_data.groupby(
            "Category"
        )["Amount"]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    st.bar_chart(
        category_expenses
    )

    st.dataframe(
        category_expenses.reset_index(),
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------
    # BUDGET PLANNING
    # --------------------------------------------------

    st.divider()

    st.subheader(
        "5. Monthly Budget Planning"
    )

    available_months = sorted(
        financial_data["Month"].unique()
    )

    selected_month = st.selectbox(
        "Select month",
        available_months
    )

    monthly_expense_data = expense_data[
        expense_data["Month"] == selected_month
    ]

    actual_category_expenses = (
        monthly_expense_data.groupby(
            "Category"
        )["Amount"]
        .sum()
    )

    categories = sorted(
        expense_data["Category"].unique()
    )

    budget_data = []

    for category in categories:

        actual_amount = actual_category_expenses.get(
            category,
            0
        )

        budget_amount = st.number_input(
            f"{category} Budget (₹)",
            min_value=0.0,
            value=float(
                max(
                    1000,
                    actual_amount * 1.2
                )
            ),
            step=500.0,
            key=f"budget_{selected_month}_{category}"
        )

        difference = (
            budget_amount - actual_amount
        )

        if actual_amount > budget_amount:

            status = "Over Budget"

        else:

            status = "Within Budget"

        budget_data.append({

            "Month": selected_month,

            "Category": category,

            "Budget": budget_amount,

            "Actual Expense": actual_amount,

            "Remaining Budget": difference,

            "Budget Status": status

        })

    budget_df = pd.DataFrame(
        budget_data
    )

    st.subheader(
        "Budget vs Actual Expense"
    )

    st.dataframe(
        budget_df,
        use_container_width=True,
        hide_index=True
    )

    over_budget_count = (
        budget_df[
            "Budget Status"
        ] == "Over Budget"
    ).sum()

    st.metric(
        "Categories Over Budget",
        int(over_budget_count)
    )


    # --------------------------------------------------
    # TRANSACTION FILTER
    # --------------------------------------------------

    st.divider()

    st.subheader(
        "6. Filter Transactions"
    )

    filter_col1, filter_col2 = st.columns(2)

    with filter_col1:

        selected_type = st.multiselect(
            "Transaction Type",
            ["Income", "Expense"],
            default=["Income", "Expense"]
        )

    with filter_col2:

        selected_category = st.multiselect(
            "Category",
            sorted(
                financial_data["Category"].unique()
            ),
            default=sorted(
                financial_data["Category"].unique()
            )
        )

    filtered_data = financial_data[
        financial_data["Type"].isin(
            selected_type
        )
        &
        financial_data["Category"].isin(
            selected_category
        )
    ]

    st.dataframe(
        filtered_data,
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------
    # MACHINE LEARNING FORECAST
    # --------------------------------------------------

    st.divider()

    st.subheader(
        "7. Next-Month Expense Estimation"
    )

    forecast, mae, error_message = (
        forecast_next_month(
            monthly_summary
        )
    )

    if error_message:

        st.warning(
            error_message
        )

    else:

        st.metric(
            "Estimated Next-Month Expenses",
            f"₹{forecast:,.2f}"
        )

        st.metric(
            "Test Mean Absolute Error",
            f"₹{mae:,.2f}"
        )

        st.caption(
            "This is a basic demonstration model. Forecast quality "
            "depends on the amount and consistency of historical "
            "data. It should not be treated as a reliable financial "
            "prediction."
        )


    # --------------------------------------------------
    # DOWNLOAD REPORT
    # --------------------------------------------------

    st.divider()

    st.subheader(
        "8. Download Financial Report"
    )

    excel_report = create_excel_report(
        financial_data,
        monthly_summary
    )

    st.download_button(
        label="Download Excel Report",
        data=excel_report,
        file_name="financial_expense_report.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        )
    )

    csv_report = financial_data.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Download CSV Report",
        data=csv_report,
        file_name="financial_expense_report.csv",
        mime="text/csv"
    )
