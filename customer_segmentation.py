# **Customer Segmentation with K-Means Clustering based on RFM Model**

This project purpose to divide customers into segments based on K-Means Clustering, which represent best marketing strategies according to the customer sales behavior, characteristics and needs.

## **Importing Required Libraries**
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from mpl_toolkits.mplot3d import Axes3D

"""## **Dataset**

The dataset obtained from database of Schneider Electric retail, contains all the transactions occurring between 01/12/2010 and 09/12/2018 for Central Asia customers.The company mainly sells electrical systems, automation and switches. Many customers of the company are distrubitors, system integrators and design companies. It can be found here and the attributes are as follows.

**InvoiceNo:** Invoice number, a 6-digit integral number uniquely assigned to each transaction.<br>
**StockCode:** Product (item) code, a 5-digit integral number uniquely assigned to each distinct product.<br>
**Description:** Product (item) name. <br>
**Quantity:** The quantities of each product (item) per transaction<br>
**InvoiceDate:** Invice Date and time, the day and time when each transaction was generated.<br>
**UnitPrice:** Unit price, Product price per unit in sterling.<br>
**CustomerID:** Customer number, a 5-digit integral number uniquely assigned to each customer.<br>
**Country:** Country name, the name of the country where each customer resides.<br>
"""

#reading the "Online_Retail" dataset
df=pd.read_excel("/content/Online Retail.xlsx")

df.head()

#exploring the data types of columns and the number of entries
df.info()

"""The dataset comprises 541909 rows and 8 columns. The data types of columns are proper to the description of columns, so there is no need to make any type and format changes.

## **Data Pre-processing**

#### **Missing Value Handling**
"""

#checking how many missing values do we have
df.isnull().sum()

"""As it is seen, there are missing values in the columns "Description" and "CustomerID". The "Description" column will not be included in further analysis, so we will handle the missing values in the "CustomerID" column by dropping the records for retail data customer segmentation."
"""

#dropping Null records
df.dropna(subset='CustomerID',inplace=True)

df['CustomerID'].isnull().sum()

df.shape

#### **Adding New Attribute "Sales" to the Dataset**
"""

#creating a column to calculate total sales value for each transaction
df['Sales'] = round(df['Quantity'] * df['UnitPrice'],2)

"""#### **Excluding return transactions**"""

#removing the negative transactions which mean return goods
df_retail=df[df["Sales"]>0]

df_retail.head()

df_retail.shape

"""#### **Reformatting column "InvoiceDate"**"""

# Extract day, month and year from InvoiceDate column into a new column InvoiceDay
df_retail['InvoiceDay'] = df_retail["Invoice date"].apply(lambda x: datetime(x.year, x.month, x.day))

df_retail.head()

"""We're all set. Now, Let's jump into it!

## **Exploratory Data Analysis**

#### **Descriptive Statistics**
"""

df_retail[["Quantity","UnitPrice"]].describe()

"""#### **What is the number of unique Customers, Stock Code, Description, Invoice No, and Invoice Date ?**"""

#finding unique numbers of attributes
df_retail.nunique()

"""We get a convenient result except for the number of StockCode and Description, which do not match each other. StockCode uniquely identifies the description of a product. In that sense, they should be equal to each other. Let's find the reason behind it."""

#finding out the StokCodes which have more than 1 Description
df_unique=df_retail.groupby(["StockCode"]).nunique().reset_index()
df_unique=df_unique[df_unique["Description"]>1][["StockCode","Description"]].sort_values(by=["Description"],ascending=False)
df_unique.head()

#selecting one of the StockCodes to analyze in detailed
df_retail[df_retail["StockCode"]==23231]["Description"].unique()

"""As a result, the discrepancy in the "Description" column won't affect our RFM analysis and can be left as it is.

#### **How many orders were cancelled?**
"""

#Finding the percentage of cancelled orders
cancelled_orders=df[df["Sales"]<0]["InvoiceNo"].nunique()
total_orders=df["InvoiceNo"].nunique()
print("Number of cancelled orders :", cancelled_orders)
print("Percentage of cancelled orders : {:.2f}%".format(cancelled_orders/total_orders*100))

"""It is a better approach to elaborate on the reason behind the cancellation of orders to provide leveraged services.

#### **What are the top 5 countries with the highest Sales?**
"""

df_top=df_retail.groupby(by=["Country"])
df_top=df_top.agg({"Sales":["sum"]})
df_top.columns = df_top.columns.droplevel(1)
df_top.sort_values(by=['Sales'],ascending=False,inplace=True)
df_top=df_top.reset_index().head(5)
df_top

fig, ax = plt.subplots(figsize=(8,5))
ax=sns.barplot(x='Country', y='Sales', data=df_top, estimator=max, ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=50, ha="right")
plt.gca().get_yaxis().set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
plt.title("Total Sales by Top 5 Countries")
plt.show()

"""Unsurprisingly, as a Kazakhstan-based  retail company, the Kazakhstan is a leader in terms of total sales.

#### **What are the top 5 countries with the highest number of customers ?**
"""

df_top=df_retail.groupby(by=["Country"])["CustomerID"].nunique()
df_top=df_top.sort_values(ascending=False).head(5)
df_top

fig, ax = plt.subplots(figsize=(8,5))
df_top.plot(kind="bar",x="Country")
ax.set_xticklabels(ax.get_xticklabels(), rotation=50, ha="right")
plt.title("Number of Customers by Top 5 Countries")
plt.show()

"""Similarly, the Kazakhstan has the highest number of customers compared to the other countries.

We are ready to continue with RFM Analysis !

## **RFM Analysis**

The first thing we are going to do is to start with **RFM (Recency, Frequency, Monetary) Analysis** , which is a customer segmentation technique for analyzing customer value based on past buying behavior and then combine our results with **K-Means Clustering Algorithm.**

#### **Feature Engineering**

The next question is what are the metrics for RFM Analysis? RFM Analysis stands for Recency, Frequency, and Monetary Analysis, and their descriptions of them are given as follows.

**• Recency:** The time since last order with the product of customers.

**• Frequency:** The total number of transaction  between the customer’s invoice date and reference day.

**• Monetary:** The total transaction value of customers.
"""

#finding the last invoice day + 1 in a new variable to calculate Recency
reference_day=df_retail["InvoiceDay"].max()+timedelta(1)

# calculating RFM values for each transaction and save them in a new dataframe "rfm"
rfm = df_retail.groupby('CustomerID').agg({
    'InvoiceDay' : lambda x: (reference_day - x.max()).days,
    'InvoiceNo' : 'count',
    'Sales' : 'sum'})
# rename the columns
rfm.rename(columns = {'InvoiceDay' : 'Recency',
                      'InvoiceNo' : 'Frequency',
                      'Sales' : 'Monetary'}, inplace = True)

rfm.head(6)

"""Here is the result table of RFM values for each customer. Now, we are going to divide customers into segments based on RFM quartiles.

#### **Customer Segmentation based on RFM Quartiles**
"""

#finding quantiles of RFM attributes
quantiles=rfm.quantile(q=[0.25,0.5,0.75])
quantiles

#assigning Recency Score from 1 to 4 to the customers
def rec_score(x):
    if x<=18:
        return 4
    elif x<=51:
        return 3
    elif x<=142.75:
        return 2
    else:
        return 1

#assigning Frequency Score from 1 to 4 to the customers
def freq_score(x):
     if x<=17:
        return 1
     elif x<=41:
        return 2
     elif x<=100:
        return 3
     else:
        return 4

#assigning Monetary Score from 1 to 4 to the customers
def mon_score(x):
      if x<=307.415:
        return 1
      elif x<=674.485:
        return 2
      elif x<=1661.74:
        return 3
      else:
        return 4

#create segmentation table
rfm_segmentation=rfm.copy()
rfm_segmentation["R"]=rfm_segmentation["Recency"].apply(rec_score)
rfm_segmentation["F"]=rfm_segmentation["Frequency"].apply(freq_score)
rfm_segmentation["M"]=rfm_segmentation["Monetary"].apply(mon_score)

rfm_segmentation.head()

"""What will happen next is we will find RFM segments and total scores to allocate the RFM groups."""

# Concatenate & sum up the three columns
rfm_segmentation['RFM_Segment'] = rfm_segmentation["R"].map(str) + rfm_segmentation["F"].map(str) + rfm_segmentation["M"].map(str)
rfm_segmentation["RFM_Score"] = rfm_segmentation[['R', 'F', 'M']].sum(axis = 1)
rfm_segmentation.head()

"""Now, it is time to end up with RFM groups. We will select 4 group names which are High Value, Loyal, At risk and Lost."""

#the distribution of RFM Groups
rfm_pie=rfm_segmentation["RFM_Group"].value_counts()
plt.pie(rfm_pie,autopct="%1.0f%%",labels=None,pctdistance=1.16,shadow=True)
plt.legend(rfm_pie.index,loc="right")

# Assigning RFM Groups based on RFM score
# rfm_labels=['Lost', 'At_risk', 'Loyal', 'High_value']
# rfm_groups=pd.qcut(rfm_segmentation["RFM_Score"],q=4,labels=rfm_labels, duplicates='drop')
# rfm_segmentation["RFM_Group"]=rfm_groups.values
# rfm_segmentation.head()

# Assigning RFM Groups based on RFM score
rfm_labels = ['Lost', 'At_risk', 'Loyal', 'High_value']

# Define bin edges manually based on the distribution of RFM scores
# Adjust the edges based on your data distribution
bin_edges = [rfm_segmentation["RFM_Score"].min(), 4, 6, 8, rfm_segmentation["RFM_Score"].max()]

# Using cut to categorize the RFM scores into the defined bins
rfm_groups = pd.cut(rfm_segmentation["RFM_Score"], bins=bin_edges, labels=rfm_labels, include_lowest=True)
rfm_segmentation["RFM_Group"] = rfm_groups.values

# Display the head of the dataframe
print(rfm_segmentation.head())

"""Lost and loyal customers make up 38% of the customer base. We can develop distinct campaigns for each targeted group. However, before deciding on a strategy, let's examine the K-Means clustering algorithm applied to the RFM values and compare the outcomes.

## **K-Means Clustering on RFM Values**

K-Means clustering is a distance-based, unsupervised machine learning algorithm. It divides data points into k clusters using the Euclidean distance metric. The algorithm is sensitive to skewness and outliers, which can distort clusters and lead to inaccurate results. To address skewness, a log transform can be applied to convert a skewed distribution to a normal or less-skewed one. Following this, normalization is necessary to ensure no single attribute disproportionately influences the clustering due to differing scales. Additionally, determining the optimal number of clusters (k) is crucial before applying K-Means. The two common methods for defining the number of clusters are:

- Elbow Method
- Silhouette Method

First, let's examine the distribution of RFM values to decide if a log transform is needed.
"""

#checking descriptive statistics
rfm.describe()

"""#### **Visualization of Distribution of RFM Values**"""

#creating histograms for each attribute
fig, ax = plt.subplots(1,3, figsize=(20,6))
sns.histplot(data=rfm,x="Recency",color="purple",bins=30,kde=True,ax=ax[0])
sns.histplot(data=rfm,x="Frequency",color="purple",bins=100,kde=True,ax=ax[1])
sns.histplot(data=rfm,x="Monetary",color="purple",bins=100,kde=True,ax=ax[2])

cols=rfm.columns[0:3]
i=0

for col in cols:
    ax[i].set_xlabel(" ")
    ax[i].set_ylabel(" ")
    ax[i].set_title(col)

    i=i+1

fig.suptitle("Histograms of Each RFM Value")

"""As it can be clearly seen that RFM attributes are highly skewed. The log transformation will be used to transform skewed data to approximately conform to normality.

#### **RFM Values Log Transform**
"""

#unskew RFM attributes with log transformations
rfm_log = rfm.apply(np.log, axis = 1).round(2)
rfm_log.head()

"""To prevent one attribute outweighs the other, let's normalize the data.

#### **RFM Values Normalization**
"""

#scaling RFM attributes
scaler=StandardScaler()
rfm_scale=scaler.fit_transform(rfm_log)

#storing into a dataframe
rfm_scale=pd.DataFrame(rfm_scale,index=rfm_log.index,columns=rfm_log.columns)
rfm_scale.head()

#creating histograms for each attribute
fig, ax = plt.subplots(1,3, figsize=(20,6))
sns.histplot(data=rfm_scale,x="Recency",color="purple",bins=30,kde=True,ax=ax[0])
sns.histplot(data=rfm_scale,x="Frequency",color="purple",bins=100,kde=True,ax=ax[1])
sns.histplot(data=rfm_scale,x="Monetary",color="purple",bins=100,kde=True,ax=ax[2])

cols=rfm.columns[0:3]
i=0

for col in cols:
    ax[i].set_xlabel(" ")
    ax[i].set_ylabel(" ")
    ax[i].set_title(col)

    i=i+1

fig.suptitle("Histograms of Each RFM Value")

#### **Determining Optimum Number of Clusters**

We will analyze two methods to find the optimum number of clusters and compare the results.

##### **Elbow Method**
"""

# find the optimum number of clusters (k) using the Elbow method
wcss = {}
for k in range(1, 8):
    kmeans = KMeans(n_clusters= k)
    kmeans.fit(rfm_scale)
    wcss[k] = kmeans.inertia_

# plot the WCSS values
sns.pointplot(x = list(wcss.keys()), y = list(wcss.values()))
plt.xlabel('K Numbers')
plt.ylabel('WCSS')
plt.show()

"""##### **Silhouette Method**"""

# find the optimum number of clusters (k) using the Silhouette method
for n_clusters in range(2,8):
    kmeans = KMeans(init='k-means++', n_clusters = n_clusters, n_init=100)
    kmeans.fit(rfm_scale)
    clusters = kmeans.predict(rfm_scale)
    silhouette_avg = silhouette_score(rfm_scale, clusters)

    print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)

"""By considering both methods, k=3 is selected as an optimum cluster number for k-means clustering algorithm.

#### **Customer Segmentation based on K-Means Clusters**
"""

# clustering
clus = KMeans(n_clusters= 3)
clus.fit(rfm_scale)

# assigning the clusters to rfm_segmentation
rfm_segmentation['K_Cluster'] = clus.labels_
rfm_segmentation.head()

"""We are almost done. Now, we will compare RFM Groups and K-Means Clusters.

## **Results**

#### **Visualization of K-Means Clusters**
"""

# joining RFM groups with K-means Clusters
rfm_scale['K_Cluster'] = clus.labels_
rfm_scale['RFM_Group'] = rfm_segmentation.RFM_Group
rfm_scale.reset_index(inplace = True)
rfm_scale.head()

# visualizing K Clusters with RFM Values
fig, ax = plt.subplots(1,3, figsize=(20,6))
sns.scatterplot(x = rfm_scale["Frequency"], y = rfm_scale["Monetary"], hue = rfm_scale["K_Cluster"],ax=ax[0])
sns.scatterplot(x = rfm_scale["Recency"], y = rfm_scale["Frequency"], hue = rfm_scale["K_Cluster"],ax=ax[1])
sns.scatterplot(x = rfm_scale["Recency"], y = rfm_scale["Monetary"], hue = rfm_scale["K_Cluster"],ax=ax[2])
fig.suptitle("Scatter Plots of K-Means Clusters")

"""#### **Visualization of RFM Groups**"""

# visualizing RFM Groups across RFM Values
fig, ax = plt.subplots(1,3, figsize=(20,6))
sns.scatterplot(x = rfm_scale["Frequency"], y = rfm_scale["Monetary"], hue = rfm_scale["RFM_Group"],ax=ax[0])
sns.scatterplot(x = rfm_scale["Recency"], y = rfm_scale["Frequency"], hue = rfm_scale["RFM_Group"],ax=ax[1])
sns.scatterplot(x = rfm_scale["Recency"], y = rfm_scale["Monetary"], hue = rfm_scale["RFM_Group"],ax=ax[2])
fig.suptitle("Scatter Plots of RFM Groups")

"""When comparing RFM groups and K-Means Clusters, we can clearly see that 3 K-Means Clusters are more distinctive. It is better to run marketing campaigns with 3 clusters and we can also take a look at a 3D scatter plot of RFM values with 3 K-clusters to get a better picture."""

# 3D scatter plot of R, F & M values with the 3 K_Clusters
fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(rfm_scale["Frequency"], rfm_scale["Monetary"], rfm_scale["Recency"], c = rfm_scale["K_Cluster"])
ax.set_xlabel("Frequency")
ax.set_ylabel("Monetary")
ax.set_zlabel("Recency")
ax.set_title("R,F,M Scores vs K-Means Clusters")

plt.show()

"""Finally, we have 3 different clusters!

## **Conclusion**

- Cluster 2 represents the high-value customers, characterized by the highest number of orders, frequency, and the most recent transactions.
- Cluster 1 consists of lost customers who rarely place orders and generate the lowest sales.
- Cluster 0 includes at-risk and loyal customers, who exhibit medium values in terms of frequency, recency, and monetary metrics.
- Across all clusters, high monetary value is associated with a high frequency of orders and more recent transactions.

## **Recommendations**

The company can design targeted marketing campaigns for various customer segments to boost revenue. To achieve this, the company might provide incentives to low-value customers to maintain their engagement and encourage more frequent purchases. Conversely, high-value customers could receive special benefits, such as exclusive discounts and early access to new products. As conclusion, the approach will depend on the company's specific business objectives.<br>
"""
