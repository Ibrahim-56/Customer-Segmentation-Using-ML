import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==========================
# LOAD DATA
# ==========================
df = pd.read_csv("customer_data.csv")

# ==========================
# DATA PREPARATION
# ==========================
features = [
    "AnnualIncome",
    "SpendingScore",
    "PurchaseFrequency"
]

X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================
# KMEANS CLUSTERING
# ==========================
kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X_scaled)

# ==========================
# BUSINESS LABELS
# ==========================
cluster_summary = df.groupby("Cluster")[features].mean()

cluster_labels = {}

for cluster in cluster_summary.index:

    income = cluster_summary.loc[cluster, "AnnualIncome"]
    spending = cluster_summary.loc[cluster, "SpendingScore"]
    frequency = cluster_summary.loc[cluster, "PurchaseFrequency"]

    if income > 80000 and spending > 60:
        cluster_labels[cluster] = "VIP Customers"

    elif spending > 75:
        cluster_labels[cluster] = "Loyal Buyers"

    elif income > 80000 and spending < 40:
        cluster_labels[cluster] = "Wealthy Low-Spenders"

    else:
        cluster_labels[cluster] = "Regular Customers"

df["Segment"] = df["Cluster"].map(cluster_labels)

# ==========================
# KPI DASHBOARD
# ==========================
print("\n========================")
print("CUSTOMER ANALYTICS REPORT")
print("========================")

print(f"Total Customers: {len(df)}")
print(f"Average Income: ${df['AnnualIncome'].mean():,.0f}")
print(f"Average Spending Score: {df['SpendingScore'].mean():.0f}")
print(f"Average Purchase Frequency: {df['PurchaseFrequency'].mean():.0f}")

# ==========================
# SEGMENT REPORT
# ==========================
segment_report = df.groupby("Segment").agg({
    "CustomerID": "count",
    "AnnualIncome": "mean",
    "SpendingScore": "mean",
    "PurchaseFrequency": "mean"
})

segment_report.columns = [
    "Customers",
    "Avg Income",
    "Avg Spending",
    "Avg Frequency"
]

print("\nSegment Summary")
print(segment_report.round(2))

# ==========================
# VISUAL STYLE
# ==========================
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10,6)

# ==========================
# 1. CUSTOMER SEGMENTS
# ==========================
plt.figure()

sns.scatterplot(
    data=df,
    x="AnnualIncome",
    y="SpendingScore",
    hue="Segment",
    palette="Set2",
    s=150
)

plt.title("Customer Segmentation Overview")
plt.tight_layout()
plt.savefig("01_customer_segments.png")
plt.show()

# ==========================
# 2. SEGMENT SIZE
# ==========================
plt.figure()

sns.countplot(
    data=df,
    x="Segment",
    palette="viridis"
)

plt.title("Customers per Segment")
plt.xticks(rotation=20)

plt.tight_layout()
plt.savefig("02_segment_size.png")
plt.show()

# ==========================
# 3. INCOME BY SEGMENT
# ==========================
plt.figure()

sns.barplot(
    data=df,
    x="Segment",
    y="AnnualIncome",
    estimator=np.mean,
    palette="rocket"
)

plt.title("Average Annual Income")
plt.xticks(rotation=20)

plt.tight_layout()
plt.savefig("03_income.png")
plt.show()

# ==========================
# 4. SPENDING SCORE
# ==========================
plt.figure()

sns.barplot(
    data=df,
    x="Segment",
    y="SpendingScore",
    estimator=np.mean,
    palette="magma"
)

plt.title("Average Spending Score")
plt.xticks(rotation=20)

plt.tight_layout()
plt.savefig("04_spending_score.png")
plt.show()

# ==========================
# 5. PURCHASE FREQUENCY
# ==========================
plt.figure()

sns.barplot(
    data=df,
    x="Segment",
    y="PurchaseFrequency",
    estimator=np.mean,
    palette="coolwarm"
)

plt.title("Average Purchase Frequency")
plt.xticks(rotation=20)

plt.tight_layout()
plt.savefig("05_purchase_frequency.png")
plt.show()

# ==========================
# 6. HEATMAP
# ==========================
segment_metrics = df.groupby("Segment")[
    [
        "AnnualIncome",
        "SpendingScore",
        "PurchaseFrequency"
    ]
].mean()

plt.figure(figsize=(8,5))

sns.heatmap(
    segment_metrics,
    annot=True,
    cmap="YlGnBu",
    fmt=".1f"
)

plt.title("Segment Characteristics Heatmap")

plt.tight_layout()
plt.savefig("06_heatmap.png")
plt.show()

# ==========================
# 7. CUSTOMER DISTRIBUTION
# ==========================
plt.figure(figsize=(12,6))

sns.boxplot(
    data=df,
    x="Segment",
    y="AnnualIncome"
)

plt.title("Income Distribution per Segment")

plt.tight_layout()
plt.savefig("07_distribution.png")
plt.show()

# ==========================
# EXECUTIVE INSIGHTS
# ==========================
print("\n========================")
print("EXECUTIVE INSIGHTS")
print("========================")

for segment in df["Segment"].unique():

    seg = df[df["Segment"] == segment]

    print(f"\n{segment}")
    print("-" * 30)

    print(
        f"Customers: {len(seg)}"
    )

    print(
        f"Average Income: "
        f"${seg['AnnualIncome'].mean():,.0f}"
    )

    print(
        f"Average Spending Score: "
        f"{seg['SpendingScore'].mean():.1f}"
    )

    print(
        f"Average Purchase Frequency: "
        f"{seg['PurchaseFrequency'].mean():.1f}"
    )

# ==========================
# EXPORT REPORT
# ==========================
segment_report.to_csv(
    "customer_segmentation_report.csv"
)

df.to_csv(
    "customer_segmentation_results.csv",
    index=False
)

print("\nFiles Generated Successfully")
print("- customer_segmentation_results.csv")
print("- customer_segmentation_report.csv")
print("- 7 Visualization Charts")
