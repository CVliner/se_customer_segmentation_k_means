# Customer Segmentation Using K-Means Clustering and Feature Engineering

## **Introduction:**
This project purpose to divide customers into segments based on K-Means Clustering, which represent best marketing strategies according to the customer sales behavior, characteristics, and needs.<br>


## **Dataset:**

The dataset obtained from database of Schneider Electric retail, contains all the transactions occurring between 01/12/2010 and 09/12/2018 for Central Asia customers. The attributes include invoice number, product code, product name, quantity per transaction, invoice date, product unit price, customerID, and country. The company mainly sells electrical systems, automation and switches. Many customers of the company are distrubitors, system integrators and design companies.

## **Recency-Frequency-Monetary (RFM) analysis to determine customer value:**

RFM (Recency, Frequency, Monetary) Analysis is a customer segmentation technique for analyzing customer value based on past buying behavior. RFM stands for the three values:

• **Recency :** The time since last order with the product of customers.<br>
• **Frequency :** The total number of transaction between the customer’s invoice date and reference day.<br>
• **Monetary :** The total transaction value of customers.<br>

After applying data pre-processing, exploratory data analysis, and feature engineering, a customer's value to a business can be quantified by considering a combination of R,F,M values. For example, a customer who has made recent high-value purchases and frequently engages in transactions is considered as high value to the business.<br>

## **Segmentaton with K-Means Clustering:**

K-Means clustering is a distance-based, unsupervised machine learning algorithm. It divides data points into k clusters using the Euclidean distance metric. The algorithm is sensitive to skewness and outliers, which can distort clusters and lead to inaccurate results. To address skewness, a log transform can be applied to convert a skewed distribution to a normal or less-skewed one. Following this, normalization is necessary to ensure no single attribute disproportionately influences the clustering due to differing scales. Additionally, determining the optimal number of clusters (k) is crucial before applying K-Means. The two common methods for defining the number of clusters are:

- Elbow Method<br>
- Silhouette Method<br>

First, let's examine the distribution of RFM values to decide if a log transform is needed.<br>

## **Results:**

<p align="center">

<img src="https://github.com/CVliner/se_customer_segmentation_k_means/blob/main/pics/Segmentation_K_Means.png" alt="cluster" width="600" height="600">

- Cluster 2 represents the high-value customers, characterized by the highest number of orders, frequency, and the most recent transactions. <br>
- Cluster 1 consists of lost customers who rarely place orders and generate the lowest sales. <br>
- Cluster 0 includes at-risk and loyal customers, who exhibit medium values in terms of frequency, recency, and monetary metrics.<br>
- Across all clusters, high monetary value is associated with a high frequency of orders and more recent transactions.<br>
 
## **Recommendations:**

The company can design targeted marketing campaigns for various customer segments to boost revenue. To achieve this, the company might provide incentives to low-value customers to maintain their engagement and encourage more frequent purchases. Conversely, high-value customers could receive special benefits, such as exclusive discounts and early access to new products. As conclusion, the approach will depend on the company's specific business objectives.<br>
 


