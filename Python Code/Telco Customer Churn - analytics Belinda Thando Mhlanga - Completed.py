# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # Telco Customer Churn Anlaysis

# COMMAND ----------

# MAGIC %md
# MAGIC ## Libraries

# COMMAND ----------

#Import libraries
#Pandas is a python library fro data analysis 
import pandas as pd

#Library that is used toperform maths calculation
import numpy as np

#Libray to plot graphs
import matplotlib.pyplot as plt

#Library to format graphs
import matplotlib.ticker as mtick

#Library to plot graphs
import seaborn as sns

#Library to ignore warnings
import warnings

warnings.filterwarnings("ignore")
 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ingest CSV

# COMMAND ----------

df = pd.read_csv("/Workspace/Users/belinda.tmhlanga@gmail.com/Analyst Lab Africa - Data Analytics Internship/Telco-Customer-Churn.csv")         

# COMMAND ----------

# MAGIC %md
# MAGIC ## EDA - Exploratory Data Analysis

# COMMAND ----------

df.columns

# COMMAND ----------

#I am looking at the number of rows and columns that exist in the code 
df.shape

print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

# COMMAND ----------

display(df) 

# COMMAND ----------

# I want to see the first 5 rows of the dataframe
df.head(5)

# COMMAND ----------


print("\nColumn names:")
print(df.columns.tolist())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Checking type of data

# COMMAND ----------

df.dtypes

# COMMAND ----------

# MAGIC %md
# MAGIC ### Checking what unique values exist in each column

# COMMAND ----------

for column in df.columns:
    print(f"\n{column}:")
    print(df[column].unique())

# COMMAND ----------

# MAGIC %md
# MAGIC ### data-cleaning 
# MAGIC - total charges to numeric

# COMMAND ----------

#TotalCharges is stored as text (`object`) instead of a number.
#We convert it to a numeric column so that we can use it in calculations and visualizations.

# Convert TotalCharges to numeric

df["TotalCharges_num"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

print("Missing values in TotalCharges after conversion:",
      df["TotalCharges_num"].isnull().sum())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Checking nulls / missing values 

# COMMAND ----------

df.isnull().sum()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Checking for duplicates in data

# COMMAND ----------

total_duplicates = df.duplicated().sum()
print("Number of total_duplicates:", total_duplicates)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Summary statistics

# COMMAND ----------

df.describe().T

# COMMAND ----------

# MAGIC %md
# MAGIC ### Understanding customer base

# COMMAND ----------

print("Total customers:", len(df))

for column in ["gender", "SeniorCitizen", "Partner", "Dependents",
               "Contract", "InternetService", "PaymentMethod"]:
    print("\n" + column)
    print(df[column].value_counts())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Customer Base - Gender

# COMMAND ----------

gender_counts = df["gender"].value_counts()

plt.figure(figsize=(6, 5))
plt.pie(
    gender_counts,
    labels=gender_counts.index,
    autopct="%1.1f%%",
    startangle=90
)
plt.title("Customer Base by Gender")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **Observation:** The customer base is fairly evenly split between male and female customers.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Customer base - Contract type

# COMMAND ----------

contract_counts = df["Contract"].value_counts()

plt.figure(figsize=(7, 5))
plt.pie(
    contract_counts,
    labels=contract_counts.index,
    autopct="%1.1f%%",
    startangle=90
)
plt.title("Customer Base by Contract Type")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **Observation:** Month-to-month contracts make up the largest group of customers.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Understanding Churn
# MAGIC - Churn = Yes means the customer left the company.
# MAGIC - Churn = No means the customer stayed

# COMMAND ----------

churn_counts = df["Churn"].value_counts()

print(churn_counts)

overall_churn_rate = (df["Churn"] == "Yes").mean() * 100

print(f"\nOverall churn rate: {overall_churn_rate:.2f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Which Segments Have the Highest Churn?
# MAGIC - We calculate the percentage of customers who churned within each segment.
# MAGIC - This is better than only counting churned customers because different segments can have different numbers of customers.

# COMMAND ----------

segments = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "InternetService",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod"
]

for segment in segments:
    print("\n" + segment)

    churn_rate = (
        pd.crosstab(
            df[segment],
            df["Churn"],
            normalize="index"
        ) * 100
    )

    print(churn_rate["Yes"].sort_values(ascending=False).round(2))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Bar chart 1 - Churn by contract type

# COMMAND ----------

churn_by_contract = (
    df.groupby("Contract")["Churn"]
    .apply(lambda x: (x == "Yes").mean() * 100)
    .sort_values(ascending=False)
)

plt.figure(figsize=(8, 5))
ax = churn_by_contract.plot(kind="bar")
plt.title("Churn Rate by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=0)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **Observation:** Month-to-month customers have a much higher churn rate than one-year and two-year customers.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Bar chart 2 - Churn by payment method

# COMMAND ----------

churn_by_payment = (
    df.groupby("PaymentMethod")["Churn"]
    .apply(lambda x: (x == "Yes").mean() * 100)
    .sort_values(ascending=False)
)

plt.figure(figsize=(10, 5))
ax = churn_by_payment.plot(kind="bar")
plt.title("Churn Rate by Payment Method")
plt.xlabel("Payment Method")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=35, ha="right")
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **Observation:** Electronic check customers have the highest churn rate among the payment methods in this dataset.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Bar chart 3 - Churn by internet service

# COMMAND ----------

churn_by_internet = (
    df.groupby("InternetService")["Churn"]
    .apply(lambda x: (x == "Yes").mean() * 100)
    .sort_values(ascending=False)
)

plt.figure(figsize=(8, 5))
ax = churn_by_internet.plot(kind="bar")
plt.title("Churn Rate by Internet Service")
plt.xlabel("Internet Service")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=0)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **Interpretation:**
# MAGIC - Fiber optic customers have the highest churn rate of the three internet service groups.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Bar chart 4 - Churn by tenure segment

# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

df["tenure_segment"] = pd.cut(
    df["tenure"],
    bins=[-1, 12, 24, 48, 72],
    labels=[
        "0-12 months",
        "13-24 months",
        "25-48 months",
        "49-72 months"
    ]
)

churn_by_tenure = (
    df.groupby("tenure_segment", observed=False)["Churn"]
    .apply(lambda x: (x == "Yes").mean() * 100)
)

plt.figure(figsize=(8, 5))
ax = churn_by_tenure.plot(kind="bar")
plt.title("Churn Rate by Tenure Segment")
plt.xlabel("Tenure")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=0)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **Observation:** Customers in their first 12 months have the highest churn rate. Churn generally decreases as customers stay longer

# COMMAND ----------

# MAGIC %md
# MAGIC ### Does Tenure Affect Loyalty?

# COMMAND ----------

plt.figure(figsize=(8, 5))
plt.hist(df["tenure"], bins=20)
plt.title("Distribution of Customer Tenure")
plt.xlabel("Tenure (months)")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **Observation:** The dataset contains many newer customers, and churn is particularly high among customers with shorter tenure.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Distribution of Monthly Charges

# COMMAND ----------

plt.figure(figsize=(8, 5))
plt.hist(df["MonthlyCharges"], bins=20)
plt.title("Distribution of Monthly Charges")
plt.xlabel("Monthly Charges")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **Observation:** Monthly charges vary considerably across customers, with many customers concentrated around the middle of the range.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Box Plot - Monthly Charges and Churn

# COMMAND ----------


plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="Churn", y="MonthlyCharges")
plt.title("Monthly Charges by Churn Status")
plt.xlabel("Churn")
plt.ylabel("Monthly Charges")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC This allows us to see whether the typical monthly charge differs between **customers** who stayed and customers who left.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Which Services Influence Churn?

# COMMAND ----------

service_columns = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies"
]

for service in service_columns:
    print("\n" + service)

    service_churn = (
        df.groupby(service)["Churn"]
        .apply(lambda x: (x == "Yes").mean() * 100)
        .sort_values(ascending=False)
    )

    print(service_churn.round(2))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Service churn chart

# COMMAND ----------

service_results = []

for service in service_columns:
    for option, group in df.groupby(service):
        churn_rate = (group["Churn"] == "Yes").mean() * 100

        service_results.append({
            "Service": service,
            "Option": option,
            "ChurnRate": churn_rate
        })

service_results_df = pd.DataFrame(service_results)

plt.figure(figsize=(12, 6))
sns.barplot(
    data=service_results_df,
    x="Service",
    y="ChurnRate",
    hue="Option"
)
plt.title("Churn Rate Across Customer Services")
plt.xlabel("Service")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **Observations:** Customers without services such as Online Security and Tech Support showhigher churn rates than customers who have these services.
# MAGIC This shows an association in the dataset; it does not prove that the service itself causes customers to stay or leave.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Correlation Heatmap
# MAGIC - Correlation shows how strongly numerical variables move together.
# MAGIC - A value close to +1 means a strong positive relationship.
# MAGIC - A value close to -1 means a strong negative relationship.
# MAGIC - A value close to 0 means little linear relationship.

# COMMAND ----------

correlation_df = df[
    ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges_num"]
].copy()

# Convert Churn to 1 = Yes and 0 = No
correlation_df["Churn"] = (df["Churn"] == "Yes").astype(int)

plt.figure(figsize=(8, 6))
sns.heatmap(
    correlation_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **Observation:**
# MAGIC - Tenure has a negative relationship with churn, meaning longer-tenure customers tend to be less likely to churn.
# MAGIC - Monthly charges have a positive relationship with churn, although this relationship is not extremely strong.
# MAGIC - Correlation does not prove that one variable causes another.

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC