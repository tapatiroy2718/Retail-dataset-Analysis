# =====================================================================
# Dataset : Retail Sales Dataset
# About   : Synthetic dataset of 4310 rows x 21 columns of order data.
# =====================================================================

# %% ===================================================================
# Phase 1 : Inspect the Raw Data
# =====================================================================

# --- Environment Setup ---
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

# --- Load dataset ---
# from google.colab import drive
# drive.mount('/content/drive')

df = pd.read_csv("/kaggle/input/datasets/satyakidas07/retail-sales-dataset/retail_sales_dataset.csv")

# --- Basic understanding regarding the dataset ---

# Display the top 5 rows
df.head()

# Display the last 5 rows
df.tail()

# Read the data from 21 to 30
df.head(31).tail(10)

help(df.iloc)

# Display first row
df.iloc[[0]]

# Display rows from 21 to 30
df.iloc[[21, 31]]
df.iloc[21:31]

df.iloc[[0, 2], [1, 2, 3, 4]]

df.loc[:, 'customer_id']

# Display 2 to 5th rows from customer_id column
df.loc[2:5, 'customer_id']

# How many rows and columns are present in this dataset?
df.shape

# Find the columns and their datatypes
df.columns
df.dtypes

# Overview of the dataset
df.info()

# Tell me about null values based on each column
df.isnull()
df.isnull().sum()

# Find unique values of all columns
columns = list(df.columns)
print(columns)
for i in columns:
    print(i, end='')
    print(df[i].unique())
    print()

df['age'].unique()

# Find the duplicate rows
df.duplicated().sum()
# Here in the dataset we have 109 duplicate rows.

df.describe()

# Summary of the dataset
# Count  : Number of non-null value
# mean   : Average value
# std    : Standard deviation (spread of data)
# min    : Smallest Value
# 25%    : First Quartile (Q1)
# 50%    : Median
# 75%    : Third Quartile
# max    : Largest Value


# %% ===================================================================
# Phase 2 : Clean and Prepare (1 to 21)
# =====================================================================

# Data Cleaning checklist
# order_id               : 30 Missing Values - Remove "ORD-" and convert data type
# order_date             : 30 Missing Value, Date Format Issue
# customer_id            : 30 missing Value, Remove "CUST" and convert data type
# customer_name          : Ensure Proper case, 30 Missing values
# age                    : 160 Missing value. Convert data type from Float to Int
# gender                 : Convert into Int by changing them into 1,2,3
# region                 : 30 Missing Value & Convert Central by C, and so on
# city                   : 30 NAN, Check city is in Proper Case
# product_category       : 30 Nan
# product_name           : 30 Null value
# quantity               : 140 NAN, Few are in -ve, Convert into Int
# unit_price             : 30 NAN, All time should be > 0
# discount_pct           : 167 NAN, All time should be > 0 and <100
# sales_amount           : 30 NAN, All time should be > 0
# profit                 : 30 NAN
# shipping_cost          : 30 NAN, All time should be > 0
# payment_method         : 30 NAN
# customer_satisfaction  : 378 NAN, Float to Int
# return_flag            : 30 NAN
# order_status           : 30 NAN, Case issues
# days_to_ship           : 130 NAN, All time should be > 0

# --- 1. order_id ---

# step 1 : remove "ORD-" from all the values in order_id column
try:
    df['order_id'] = df['order_id'].str.replace("ORD-0", "", regex=False)
    display(df.head())
except Exception:
    print("Already removed")

# step 2: drop rows where duplicate order_id present
df = df.drop_duplicates(subset='order_id')
display(df.head())
df['order_id'].describe()

# step 3: remove NaN
df['order_id'].isnull().sum()
df[df['order_id'].isna()]
df = df.dropna(subset=['order_id'])
df[df['order_id'].isna()]

# step 4 : change the data type
df['order_id'].dtypes
df['order_id'] = df['order_id'].astype(int)
df['order_id'].dtypes

# --- Data cleaning - order_date ---
df['order_date'].unique()

# convert all order_date to 'YYYY-MM-DD' format
df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce').dt.strftime('%y-%m-%d')
df['order_date'].dtypes

# convert all order_date to date format
df['order_date'] = pd.to_datetime(df['order_date'])
df['order_date'].dtypes

# count the NaN values in order_date column
df['order_date'].isnull().sum()

# check the frequency of order_date
df['order_date'].value_counts().sort_values()

# replace NaN value of order_date by '2023-01-12'
df['order_date'] = df['order_date'].fillna(pd.Timestamp('2023-01-12'))
df['order_date'].isnull().sum()

df.isnull().sum()

# --- customer_id: 30 missing values, remove "CUST" and convert data type ---

# step 1 : remove "CUST" from all the values in customer_id column
try:
    df['customer_id'] = df['customer_id'].str.replace("CUST", "", regex=False)
    display(df.head())
except Exception:
    print("Already Removed")

# step 2: count and remove the NaN values
df['customer_id'].isnull().sum()
df = df.dropna(subset=['customer_id'])
df['customer_id'].isnull().sum()

# step 3 : convert the data type
df['customer_id'] = df['customer_id'].astype(int)
df['customer_id'].dtypes

# --- 4. customer_name: ensure Proper case, 30 missing values ---

# step 1: count the NaN values
df['customer_name'].isnull().sum()

# step 2: replace NaN with 'unknown' and ensure proper case
df['customer_name'] = df['customer_name'].fillna('unknown')
df['customer_name'] = df['customer_name'].str.title()
df['customer_name'].unique()[:10]

# --- age: 160 missing values, convert data type from Float to Int ---

# step 1: check the data distribution
df['age'].describe()

# step 2: replace invalids with NaN
df.loc[(df['age'] < 0) | (df['age'] > 100), 'age'] = np.nan
df['age'].describe()
df['age'].isnull().sum()

# step 3: replace NaN by median
median_age = df['age'].median()
df['age'] = df['age'].fillna(median_age)
df['age'].isnull().sum()

# --- gender: convert into int by changing them into 1,2,3 ---

# step 1: find out the unique values
df['gender'].unique()

df['gender'] = df['gender'].str.upper()
df['gender'].unique()

gender_map = {'MALE': 1, 'FEMALE': 2, 'OTHER': 3, 'M': 1, 'F': 2}
df['gender'] = df['gender'].map(gender_map).astype(int)
df['gender'].unique()

# --- region: 30 missing values & convert Central by C, and so on ---

# step 1: check unique and null
df['region'].unique()

# step 2: convert long text to short
region_map = {'Central': 'C', 'East': 'E', 'West': 'W', 'North': 'N', 'South': 'S'}
df['region'] = df['region'].map(region_map)
df['region'].unique()

# --- city: 30 NaN, check city is in proper case ---
df['city'].unique()
df['city'].isnull().sum()

# --- product_category: 30 NaN ---
df['product_category'].isnull().sum()
df['product_category'].unique()
df['product_category'].mode()[0]

# --- product_name: 30 null values ---
df['product_name'].unique()

# --- quantity: 140 NaN, few are negative, convert into int ---

# step 1: check the range (some values are negative)
df['quantity'].describe()
df['quantity'].unique()

df.loc[(df['quantity'] < 0), 'quantity'] = np.nan
df['quantity'].describe()

df['quantity'] = df['quantity'].fillna(df['quantity'].median())
df['quantity'] = df['quantity'].astype(int)
df['quantity'].dtypes

# --- unit_price: 30 NaN, all time should be > 0 ---
df['unit_price'].isnull().sum()
df['unit_price'].describe()

# --- discount_pct: 167 NaN, all time should be > 0 and <100 ---
df['discount_pct'].isnull().sum()
df['discount_pct'].describe()
df['discount_pct'].median()

df['discount_pct'] = df['discount_pct'].fillna(df['discount_pct'].median())
df['discount_pct'].describe()

# --- sales_amount: 30 NaN, all time should be > 0 ---
df['sales_amount'].isnull().sum()

# --- profit: 30 NaN ---
df['profit'].isnull().sum()
df['profit'].describe()

# --- shipping_cost: 30 NaN, all time should be > 0 ---
df['shipping_cost'].describe()

df.loc[(df['shipping_cost'] < 0), 'shipping_cost'] = np.nan
df['shipping_cost'].describe()
df['shipping_cost'].isnull().sum()

df['shipping_cost'] = df['shipping_cost'].fillna(df['shipping_cost'].median())
df['shipping_cost'].describe()

# --- payment_method: 30 NaN ---
df['payment_method'].isnull().sum()
df['payment_method'].unique()

# --- customer_satisfaction: 378 NaN, Float to Int ---
df['customer_satisfaction'].isnull().sum()
df['customer_satisfaction'].describe()
# NOTE: left as-is (with NaNs) -- satisfaction is not imputed since a rating
# that was never given shouldn't be guessed at with a fill value.

# --- return_flag: 30 NaN ---

# step 1: check missing values
df['return_flag'].isnull().sum()

# step 2: check unique values
df['return_flag'].unique()

# step 3: verify the datatype
df['return_flag'].dtype

# step 4: convert to boolean
df['return_flag'] = df['return_flag'].astype(bool)

# step 5: verify
df['return_flag'].dtype

# step 6: final validation
df['return_flag'].unique()

# --- order_status: 30 NaN, case issues ---

# step 1: check missing values
df['order_status'].isnull().sum()

# step 2: check unique values
df['order_status'].unique()

# step 3: fix case issues
df['order_status'] = df['order_status'].str.title()

# step 4: verify
df['order_status'].unique()

# step 5: verify missing values again
df['order_status'].isnull().sum()

# step 6: check datatype
df['order_status'].dtype

# --- days_to_ship: 130 NaN, all time should be > 0 ---

# step 1: check missing values
df['days_to_ship'].isnull().sum()

# step 2: check statistics
df['days_to_ship'].describe()

# step 3: count invalid values
(df['days_to_ship'] <= 0).sum()

# step 4: replace invalid values with NaN
df.loc[df['days_to_ship'] <= 0, 'days_to_ship'] = np.nan

# step 5: check missing values again
df['days_to_ship'].isnull().sum()

# step 6: fill missing values with median
df['days_to_ship'] = df['days_to_ship'].fillna(df['days_to_ship'].median())

# step 7: verify missing values
df['days_to_ship'].isnull().sum()

# step 8: convert float to integer
df['days_to_ship'] = df['days_to_ship'].astype(int)

# step 9: final verification
df['days_to_ship'].dtype
df['days_to_ship'].describe()

# --- Final Dataset Validation ---
df.info()
df.isnull().sum()
df.describe()


# %% ===================================================================
# Phase 3 : Analyze
# Explore relationships and statistically test them.
# =====================================================================

# Numerical   : age, quantity, unit_price, discount_pct, sales_amount,
#               profit, shipping_cost, customer_satisfaction, days_to_ship
# Categorical : gender, region, city, product_category, product_name,
#               payment_method, order_status, return_flag

# --- Step 1 : Univariate Analysis ---

num_cols = ["age", "quantity", "unit_price", "discount_pct",
            "sales_amount", "profit", "shipping_cost",
            "customer_satisfaction", "days_to_ship"]

fig, axes = plt.subplots(3, 3, figsize=(16, 11))
for ax, c in zip(axes.flat, num_cols):
    sns.histplot(df[c].dropna(), kde=True, ax=ax, color="#3b6fa0")
    ax.set_title(f"Distribution of {c}")
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(3, 3, figsize=(16, 10))
for ax, c in zip(axes.flat, num_cols):
    sns.boxplot(x=df[c], ax=ax, color="#7fa8d9")
    ax.set_title(f"Boxplot of {c}")
plt.tight_layout()
plt.show()

summary_stats = df[num_cols].agg(["mean", "median", "std",
                                   lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan,
                                   "skew", "kurt"])
summary_stats.index = ["mean", "median", "std", "mode", "skew", "kurtosis"]
summary_stats

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
sns.countplot(data=df, x="region", order=df["region"].value_counts().index, ax=axes[0], color="#3b6fa0")
axes[0].set_title("Orders by Region")
axes[0].tick_params(axis="x", rotation=30)

sns.countplot(data=df, x="product_category", order=df["product_category"].value_counts().index, ax=axes[1], color="#5f8fc7")
axes[1].set_title("Orders by Product Category")
axes[1].tick_params(axis="x", rotation=30)

df["order_status"].value_counts().plot.pie(autopct="%1.1f%%", ax=axes[2], ylabel="")
axes[2].set_title("Order Status Split")
plt.tight_layout()
plt.show()


# --- Step 2 : Bivariate Analysis ---

# Numerical vs Numerical
fig, ax = plt.subplots(figsize=(7, 5))
sns.scatterplot(data=df, x="unit_price", y="sales_amount", alpha=0.3, ax=ax, color="#3b6fa0")
ax.set_title("Sales Amount vs Unit Price")
plt.show()

pearson_r, pearson_p = stats.pearsonr(df["unit_price"], df["sales_amount"])
print(f"Pearson r = {pearson_r:.3f}  (p = {pearson_p:.2e})")

# Numerical vs Categorical
fig, ax = plt.subplots(figsize=(9, 5))
sns.boxplot(data=df, x="product_category", y="sales_amount", hue="product_category", legend=False, ax=ax, palette="Blues")
ax.set_title("Sales Amount by Product Category")
plt.xticks(rotation=20)
plt.show()

fig, ax = plt.subplots(figsize=(8, 5))
df.groupby("region")["profit"].mean().sort_values().plot.barh(ax=ax, color="#3b6fa0")
ax.set_title("Average Profit by Region")
plt.show()

# Categorical vs Categorical
ct = pd.crosstab(df["region"], df["order_status"], normalize="index") * 100
ct.plot(kind="bar", stacked=True, figsize=(9, 5), colormap="Blues_r")
plt.title("Order Status Composition by Region (%)")
plt.ylabel("% of orders")
plt.xticks(rotation=0)
plt.legend(title="Order Status", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()
ct.round(1)


# --- Step 20 : Multivariate Analysis ---

sample_df = df.sample(min(600, len(df)), random_state=42)
sns.pairplot(sample_df[["quantity", "unit_price", "sales_amount", "profit"]].dropna(),
             diag_kind="kde", plot_kws={"alpha": 0.4, "s": 15, "color": "#3b6fa0"})
plt.show()

corr = df[num_cols].corr(numeric_only=True)
fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", ax=ax)
ax.set_title("Correlation Heatmap — Numerical Columns")
plt.show()

fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(data=df, x="region", y="sales_amount", hue="payment_method", ax=ax, palette="Blues")
ax.set_title("Sales Amount by Region, grouped by Payment Method")
plt.xticks(rotation=0)
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()

g = sns.FacetGrid(df, col="region", col_wrap=3, height=3)
g.map_dataframe(sns.histplot, x="profit", color="#3b6fa0", bins=20)
g.set_titles("{col_name}")
g.fig.suptitle("Profit Distribution by Region", y=1.03)
plt.show()


# --- Step 21 : Hypothesis Testing ---

# 21a — Pearson correlation (unit_price vs sales_amount)
r, p = stats.pearsonr(df["unit_price"], df["sales_amount"])
print(f"Pearson r = {r:.3f}, p-value = {p:.2e}")
print("Reject H0 -> significant correlation" if p < 0.05 else "Fail to reject H0")

# 21b — t-test (sales_amount: returned vs not returned orders)
returned = df.loc[df["return_flag"] == True, "sales_amount"].dropna()
not_returned = df.loc[df["return_flag"] == False, "sales_amount"].dropna()

shapiro_ret = stats.shapiro(returned.sample(min(500, len(returned)), random_state=1))
levene_stat, levene_p = stats.levene(returned, not_returned)
print(f"Shapiro-Wilk (returned sample) p = {shapiro_ret.pvalue:.4f}")
print(f"Levene's test p = {levene_p:.4f}")

t_stat, t_p = stats.ttest_ind(returned, not_returned, equal_var=(levene_p > 0.05))
print(f"t-statistic = {t_stat:.3f}, p-value = {t_p:.4f}")
print("Reject H0 -> sales amount differs by return status" if t_p < 0.05 else "Fail to reject H0")

# 21c — One-way ANOVA
groups = [g["profit"].dropna().values for _, g in df.groupby("product_category")]
levene_stat, levene_p = stats.levene(*groups)
f_stat, anova_p = stats.f_oneway(*groups)

print(f"Levene's test p = {levene_p:.4f}")
print(f"ANOVA F = {f_stat:.2f}, p-value = {anova_p:.2e}")
print("Reject H0 -> profit differs by product category" if anova_p < 0.05 else "Fail to reject H0")

df.groupby("product_category")["profit"].mean().round(1)

# 21d — Chi-square test (region vs order_status)
contingency = pd.crosstab(df["region"], df["order_status"])
chi2, chi_p, dof, expected = stats.chi2_contingency(contingency)

print(f"Chi-square = {chi2:.2f}, dof = {dof}, p-value = {chi_p:.4f}")
print("Reject H0 -> region and order_status are associated" if chi_p < 0.05 else "Fail to reject H0")
contingency
